"""Paired GT-observation action evaluation, matching the step1000 protocol.

Teacher-forced and generated subtask conditions share the action noise seed.
No environment rollout is used here; closed-loop success uses eval_success.
"""
from pathlib import Path
import argparse, datetime, json, os
import numpy as np
from simplememvla.compat import apply_fla_torch_compat
apply_fla_torch_compat()
import torch
import yaml
from transformers import AutoProcessor
from configs.sft_params import DataArguments
from train import resolve_benchmark_defaults
from simplememvla.data.dataset import SimpleMemVLADataset
from simplememvla.data.collator import SimpleMemVLADataCollator, pad_memory
from simplememvla.model.modeling_simplememvla import sample_flow_timesteps
from robomme_sim.policy import get_vla, RoboMMEPolicy
from robomme_sim.batched_policy import BatchedEvalPolicy
from .kinematics import PandaFK
from .metrics import error_arrays, summarize_metrics

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def scalar(x):
    return int(torch.as_tensor(x).reshape(-1)[0])

def arr(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
    return np.asarray(x)

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

class GTObservationPolicy(BatchedEvalPolicy):

    def _action_memory_kwargs(self, processed_list):
        if not self.model.config.use_memory:
            return {}
        samples = [p['diag_sample'] for p in processed_list]
        mem = pad_memory([s['mem_feats'] for s in samples], [s['mem_age'] for s in samples])
        mem = {k: v.to(self.device) for (k, v) in mem.items()}
        (c, t, m) = self.model._memory_inputs(mem['mem_feats'], mem['mem_age'], mem['mem_mask'], self.device)
        return dict(memory_cond=c, memory_tokens=t, memory_attention_mask=m)

def instruction_from_meta(meta, index):
    table = meta.tasks
    if 'task_index' in table.columns:
        row = table[table.task_index == index].iloc[0]
    else:
        row = table.iloc[index]
    return str(row['task'] if 'task' in row.index else row.name)

def prepare_dataset(cfg, checkpoint, episodes, args):
    a = DataArguments(benchmark='robomme', root=str(args.dataset_root), episodes=episodes, stats_path=str(checkpoint / 'stats.json'), single_frame=False, current_frame_only=True, image_aug=False, variable_history=False, mem_feature_dir=str(args.feature_dir) if cfg.use_memory else None, mem_window_sec=cfg.mem_window_sec, mem_fps=cfg.mem_fps)
    spec = resolve_benchmark_defaults(a)
    return SimpleMemVLADataset(a, spec, condition_source='subtask', mem_frame_pool=cfg.mem_frame_pool if cfg.use_memory else None)

@torch.no_grad()
def openloop(args, model, proc, unnorm, norm_state, output, fk, ids):
    episodes = sorted((args.episode_bases[t] + e for (t, eps) in ids.items() for e in eps))
    ds = prepare_dataset(model.config, args.checkpoint, episodes, args)
    hf = ds.dataset.reader.hf_dataset
    ep = np.array([scalar(x) for x in hf['episode_index']])
    frame = np.array([scalar(x) for x in hf['frame_index']])
    raw_states = np.stack([arr(x).reshape(-1) for x in hf['observation.state']]).astype(np.float32)
    raw_actions = np.stack([arr(x).reshape(-1) for x in hf['action']]).astype(np.float32)
    anchors = np.asarray(ds._anchor_indices if ds._anchor_indices is not None else np.arange(len(ds)), dtype=int)
    starts = {int(e): int(frame[anchors[ep[anchors] == e]].min()) for e in np.unique(ep)}
    task_for_ep = {args.episode_bases[t] + e: t for (t, eps) in ids.items() for e in eps}
    selected = []
    for (didx, ridx) in enumerate(anchors):
        e = int(ep[ridx])
        f = int(frame[ridx])
        start = starts[e]
        if (f - start) % args.stride == 0 and f - start < args.max_steps:
            selected.append((didx, int(ridx), e, f, task_for_ep[e]))
    if args.max_samples:
        selected = selected[:args.max_samples]
    print('[openloop] anchors=', len(selected), 'episodes=', len(episodes), flush=True)
    save_json(output / 'sampling.json', {'anchors': len(selected), 'episodes': episodes, 'start_frames': starts, 'stride': args.stride, 'anchor_frames': [dict(dataset_index=x[0], row_index=x[1], episode=x[2], frame=x[3], task=x[4]) for x in selected]})
    collator = SimpleMemVLADataCollator(proc, max_length=8192, condition_source='subtask')
    policy = GTObservationPolicy(model, proc, unnorm, norm_state, device=torch.device('cuda:0'), num_denoising_steps=args.num_denoising_steps, eval_temperature=args.eval_temperature, max_subtask_tokens=args.max_subtask_tokens)
    mean = arr(unnorm.stats['mean'])
    std = np.maximum(arr(unnorm.stats['std']), 0.01)
    collected = {k: [] for k in ['gt', 'tf', 'gen', 'state', 'valid', 'flow_mse', 'flow_mse_dim', 'zero_flow_mse', 'one_step_clean_mse', 'timesteps', 'episode', 'frame']}
    meta = []
    with open(output / 'per_prediction.jsonl', 'w', buffering=1) as log:
        for (bi, start) in enumerate(range(0, len(selected), args.batch_size)):
            rows = selected[start:start + args.batch_size]
            samples = [ds[x[0]] for x in rows]
            batch = collator(samples)
            inp = {k: v.to('cuda:0') for (k, v) in batch.items()}
            mm = {k: inp[k] for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'mm_token_type_ids'] if k in inp}
            pos = model._position_ids(inp['input_ids'], inp['attention_mask'], mm)
            (hidden, ce) = model._backbone_forward(inp['input_ids'], inp['attention_mask'], position_ids=pos, labels=inp['labels'], **mm)
            (ch, ct, cm, cp) = model._gather_condition(hidden, inp['input_ids'], inp['labels'].ne(-100), pos, inp['attention_mask'])
            (mc, mt, mk) = model._memory_inputs(inp.get('mem_feats'), inp.get('mem_age'), inp.get('mem_mask'), ch.device)
            conditions = dict(condition_position_ids=cp, condition_attention_mask=cm, condition_token_embeds=ct, state=inp['state'], memory_cond=mc, memory_tokens=mt, memory_attention_mask=mk)
            actions = inp['actions'].to(ch.dtype)
            mask = inp['action_mask']
            torch.manual_seed(args.flow_seed + bi)
            noise = torch.randn_like(actions)
            ts = sample_flow_timesteps(len(rows), ch.device, ch.dtype, beta_alpha=model.config.timestep_beta_alpha, beta_beta=model.config.timestep_beta_beta)
            xt = (1 - ts[:, None, None]) * actions + ts[:, None, None] * noise
            target = noise - actions
            velocity = model.action_head(ch, xt, ts, **conditions)
            ve = (velocity.float() - target.float()).square()
            den = mask.sum(1).clamp_min(1)
            flowdim = (ve * mask).sum(1) / den
            flow = flowdim.mean(1)
            zero = ((target.float().square() * mask).sum(1) / den).mean(1)
            one = ((ve * ts[:, None, None].float().square() * mask).sum(1) / den).mean(1)
            if bi == 0:
                full = model(**inp, noise=noise, timesteps=ts)
                assert torch.allclose(flow.mean(), full.action_loss.float(), atol=0.0002, rtol=0.02), (flow.mean(), full.action_loss)
                save_json(output / 'loss_path_check.json', {'manual_flow_loss': float(flow.mean()), 'model_forward_action_loss': float(full.action_loss), 'matched': True})
            torch.manual_seed(args.action_seed + bi)
            tf = model.action_head.sample(ch, num_steps=args.num_denoising_steps, temperature=args.eval_temperature, **conditions)
            gt = arr(unnorm.unnormalize(batch['actions'].float()))
            tfraw = arr(unnorm.unnormalize(tf.float().cpu()))
            state = arr(ds.state_normalizer.unnormalize(batch['state'].float()))
            valid = arr(batch['action_mask'])[:, :, 0].astype(bool)
            processed = []
            for (sample, row) in zip(samples, rows):
                record = hf[row[1]]
                instruction = instruction_from_meta(ds.ds_meta, scalar(record['task_index']))
                buffer = RoboMMEPolicy(model, proc, unnorm, norm_state, current_frame_only=True)
                videos = iter(sample['videos'])
                images = iter(sample['images'])
                frames = {}
                for key in model.config.image_keys:
                    frames[key] = next(videos)[-1] if key in model.config.history_image_keys else next(images)
                buffer.observe(frames)
                p = buffer._prepare_inputs(instruction)
                p['diag_sample'] = sample
                processed.append(p)
            if bi == 0:
                prefix = processed[0]['input_ids'][0].cpu()
                trainprefix = batch['input_ids'][0, :len(prefix)].cpu()
                firstkey = processed[0]['video_patch_keys'][0]
                pixels = processed[0]['video_patch_pixels'][firstkey].cpu()
                parity = {'prefix_ids_match': bool(torch.equal(prefix, trainprefix)), 'prefix_length': len(prefix), 'video_pixels_match': bool(torch.equal(batch['pixel_values_videos'][:len(pixels)], pixels))}
                save_json(output / 'train_infer_input_parity.json', parity)
                assert parity['video_pixels_match'], parity
                assert parity['prefix_ids_match'], parity
            captured = {}
            original_sample = model.action_head.sample

            def capture_condition(hidden, *aa, **kk):
                captured.update(hidden=hidden.detach().clone(), token=kk['condition_token_embeds'].detach().clone(), mask=kk['condition_attention_mask'].detach().clone(), position=kk['condition_position_ids'].detach().clone())
                return original_sample(hidden, *aa, **kk)
            model.action_head.sample = capture_condition
            torch.manual_seed(args.action_seed + bi)
            generated = policy.generate_batch(processed, [s['state'].unsqueeze(0).to('cuda:0') for s in samples])
            model.action_head.sample = original_sample
            gen = np.stack([x[0] for x in generated])
            errors = {mode: error_arrays(pred, gt, state, valid, fk, mean, std) for (mode, pred) in [('teacher_forced', tfraw), ('generated', gen)]}
            for (i, row) in enumerate(rows):
                (didx, ridx, e, f, task) = row
                v = int(valid[i].sum())
                assert np.allclose(gt[i, :v], raw_actions[ridx:ridx + v], atol=2e-06), (e, f, 'action misalignment')
                assert np.allclose(state[i], raw_states[ridx], atol=2e-06)
                gttext = ds._get_subtask(hf[ridx])
                item = {'index': start + i, 'task': task, 'episode': e, 'frame': f, 'elapsed_control_steps': f - starts[e], 'valid_horizon': v, 'gt_subtask': gttext, 'generated_subtask': generated[i][1], 'exact_subtask_match': gttext.strip() == generated[i][1].strip(), 'flow_velocity_mse': float(flow[i]), 'flow_mse_per_dimension': arr(flowdim[i]).tolist(), 'one_step_clean_normalized_mse': float(one[i]), 'noise_seed': args.action_seed + bi, 'teacher_forced_subtask_ce': float(model._sparse_ce_loss(hidden[i:i + 1], inp['labels'][i:i + 1])), 'modes': {}}
                nt = int(cm[i].sum())
                ng = int(captured['mask'][i].sum())
                same_tokens = nt == ng and bool(torch.equal(ct[i, :nt], captured['token'][i, :ng]))
                item['same_condition_tokens'] = same_tokens
                if same_tokens:
                    th = ch[i, :nt].float()
                    gh = captured['hidden'][i, :ng].float()
                    item['same_token_hidden_relative_l2'] = float((th - gh).norm() / th.norm().clamp_min(1e-09))
                    item['same_token_hidden_cosine'] = float(torch.nn.functional.cosine_similarity(th, gh, dim=-1).mean())
                    item['same_condition_positions'] = bool(torch.equal(cp[:, i, :nt], captured['position'][:, i, :ng]))
                for mode in errors:
                    er = errors[mode]
                    m = valid[i, :16]
                    item['modes'][mode] = {'first16_joint_mae_rad': float(er['joint_abs'][i, :16][m].mean()), 'first16_tcp_mean_mm': float(er['tcp_mm'][i, :16][m].mean()), 'joint_mae_by_horizon_rad': er['joint_abs'][i].mean(-1).tolist(), 'tcp_error_by_horizon_mm': er['tcp_mm'][i].tolist(), 'gripper_sign_accuracy': float(1 - er['gripper_wrong'][i, :16][m].mean())}
                item['gt_first16_motion_rad'] = float(np.abs(gt[i, :16, :7] - state[i, None, :7])[valid[i, :16]].mean())
                log.write(json.dumps(item, ensure_ascii=False) + '\n')
                meta.append(item)
            for (key, value) in [('gt', gt), ('tf', tfraw), ('gen', gen), ('state', state), ('valid', valid), ('flow_mse', arr(flow)), ('flow_mse_dim', arr(flowdim)), ('zero_flow_mse', arr(zero)), ('one_step_clean_mse', arr(one)), ('timesteps', arr(ts)), ('episode', np.array([x[2] for x in rows])), ('frame', np.array([x[3] for x in rows]))]:
                collected[key].append(value)
            if bi % 10 == 0 or start + len(rows) == len(selected):
                print('[openloop]', start + len(rows), '/', len(selected), 'flow=', float(flow.mean()), flush=True)
                save_json(output / 'progress.json', {'state': 'openloop', 'done': start + len(rows), 'total': len(selected), 'time': now()})
    raw = {k: np.concatenate(v, axis=0) for (k, v) in collected.items()}
    np.savez_compressed(output / 'predictions.npz', **raw)
    summary = {'same_condition_token_rate': float(np.mean([m['same_condition_tokens'] for m in meta])), 'teacher_forced_subtask_ce': float(np.mean([m['teacher_forced_subtask_ce'] for m in meta])), 'samples': len(meta), 'action_std': std.tolist(), 'action_mean': mean.tolist(), 'overall': summarize_metrics(raw, fk, mean, std), 'per_task': {}, 'exact_generated_subtask_match_rate': float(np.mean([x['exact_subtask_match'] for x in meta]))}
    for task in ids:
        keep = np.array([m['task'] == task for m in meta])
        if keep.any():
            summary['per_task'][task] = summarize_metrics({k: v[keep] for (k, v) in raw.items()}, fk, mean, std)
    for (title, keep) in [('motion_lt_0.01', [m['gt_first16_motion_rad'] < 0.01 for m in meta]), ('motion_0.01_0.03', [0.01 <= m['gt_first16_motion_rad'] < 0.03 for m in meta]), ('motion_ge_0.03', [m['gt_first16_motion_rad'] >= 0.03 for m in meta])]:
        keep = np.array(keep, bool)
        if keep.any():
            summary.setdefault('motion_buckets', {})[title] = {'samples': int(keep.sum()), 'metrics': summarize_metrics({k: v[keep] for (k, v) in raw.items()}, fk, mean, std)}
    save_json(output / 'openloop_summary.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-json', dest='episodes_json', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=0, help='Smoke only; 0 evaluates all anchors')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    for key in ['batch_size','stride','max_steps','num_denoising_steps','eval_temperature','max_subtask_tokens','flow_seed','action_seed','episode_bases']:
        setattr(args, key, protocol[key])
    args.dataset_root = Path(protocol['dataset_root'])
    args.feature_dir = Path(protocol['feature_dir'])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'predictions.npz').exists():
        raise FileExistsError('Completed predictions already exist: '+str(args.output))
    save_json(args.output/'status.json', {'state':'loading','started_at':now(),'experiment':args.experiment,'pid':os.getpid()})
    torch.set_num_threads(2)
    fk = PandaFK()
    model, unnorm, norm_state = get_vla(str(args.checkpoint), 'bfloat16', attn_implementation='flash_attention_2')
    model.requires_grad_(False)
    model.eval()
    assert model.config.current_frame_only and model.config.condition_source == 'subtask'
    assert model.config.action_horizon == 30, 'Protocol v1 expects full 30-step chunks'
    proc = AutoProcessor.from_pretrained(args.checkpoint, padding_side='right')
    ids = json.loads(args.episodes_json.read_text())
    summary = openloop(args, model, proc, unnorm, norm_state, args.output, fk, ids)
    save_json(args.output/'status.json', {'state':'complete','ended_at':now(),'experiment':args.experiment,'pid':os.getpid(),'samples':summary['samples']})
    print('ACTION_ERRORS_COMPLETE',args.experiment,flush=True)

if __name__ == '__main__':
    main()
