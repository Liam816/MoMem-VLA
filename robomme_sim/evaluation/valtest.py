"""Config-driven closed-loop held-out val40 + official test200; no model changes."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid
from zoneinfo import ZoneInfo

import yaml
from simplememvla.config import load_config

REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def read_episode_set(path):
    values = [int(x) for x in Path(path).read_text().replace(',', ' ').split()]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate episodes: {path}')
    return set(values)


def validate_result(path, episode_ids):
    """The legacy evaluator can exit zero with excluded tasks; reject that."""
    result = json.loads(path.read_text())
    if result.get('n_excluded_na') != 0 or set(result.get('tasks', {})) != set(episode_ids):
        raise ValueError(f'Missing/excluded tasks: {path}')
    rows = {}; total = 0; successes = 0
    for task, ids in episode_ids.items():
        row = result['tasks'][task]; rate = row.get('success_rate')
        if row.get('n') != len(ids) or row.get('oom') is not False:
            raise ValueError(f'Incomplete/OOM result {task}: {row}')
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f'Invalid success rate {task}: {row}')
        count = round(rate * len(ids))
        if abs(rate * len(ids) - count) > 1e-6:
            raise ValueError(f'Rate is inconsistent with episode count: {row}')
        rows[task] = {'successes': count, 'episodes': len(ids), 'success_rate': rate}
        successes += count; total += len(ids)
    macro = sum(x['success_rate'] for x in rows.values()) / len(rows)
    if not math.isclose(result.get('macro_avg', float('nan')), macro, abs_tol=1e-8):
        raise ValueError(f'Invalid aggregate rate: {path}')
    return {'tasks': rows, 'successes': successes, 'episodes': total,
            'success_rate': successes / total, 'macro_avg': macro}


def validate_eval_args(directory, checkpoint, protocol):
    actual = json.loads((directory / 'eval_args.json').read_text())
    if Path(actual['pretrained_checkpoint']).resolve() != checkpoint:
        raise ValueError('Evaluator loaded an unexpected checkpoint')
    if actual.get('num_gpus') != 1 or actual.get('backbone_override') is not None:
        raise ValueError('Unexpected worker count or backbone override')
    if Path(actual['episode_ids_json']).resolve() != directory / 'episode_ids.json':
        raise ValueError('Evaluator used an unexpected episode manifest')
    for key, expected in protocol.items():
        if actual.get(key) != expected:
            raise ValueError(f'Evaluator argument mismatch {key}: {actual.get(key)!r} != {expected!r}')


def prepare(config_path, selected):
    config = load_config(config_path)
    if config.get('schema_version') != 1:
        raise ValueError('Expected schema_version: 1')
    checkpoint = repo_path(config['checkpoint'])
    model = json.loads((checkpoint / 'config.json').read_text())
    if not model.get('current_frame_only') or model.get('condition_source') != 'subtask':
        raise ValueError('This suite requires a current-frame-only, subtask-conditioned checkpoint')
    if model.get('use_memory') and (model.get('mem_arch') != 'naive_tokens' or model.get('mem_fusion') != 'prefix_tokens'):
        raise ValueError('Expected naive_tokens memory -> DiT prefix')
    if model.get('action_horizon') != 30:
        raise ValueError('Expected action_horizon=30')
    weights = checkpoint / 'model.safetensors'
    if not weights.is_file() or weights.stat().st_size < 1_000_000_000:
        raise ValueError(f'Full HF model.safetensors required: {checkpoint}')
    trainer_path = checkpoint / 'trainer_state.json'
    trainer = json.loads(trainer_path.read_text()) if trainer_path.exists() else {}
    if checkpoint.name.startswith('checkpoint-'):
        expected = int(checkpoint.name.removeprefix('checkpoint-'))
        if trainer.get('global_step') != expected:
            raise ValueError('Checkpoint save incomplete or directory step does not match trainer_state')
    tasks = config['tasks']
    if len(tasks) != 4 or len(set(tasks)) != 4:
        raise ValueError('This protocol evaluates the four trained tasks')
    protocol = config['protocol']
    if protocol.get('current_frame_only') is not True or protocol.get('pipelined') is not False:
        raise ValueError('Require single-frame, non-pipelined inference')
    if protocol.get('fail_on_env_error') is not True:
        raise ValueError('Environment errors must fail evaluation')
    unknown = set(selected or []) - set(config['evaluations'])
    if unknown:
        raise ValueError(f'Unknown evaluation split(s): {sorted(unknown)}')
    selections = selected or list(config['evaluations'])
    stages = {}; metadata_hashes = {}
    for name in selections:
        if name not in {'val40', 'test200'}:
            raise ValueError(f'Unknown protocol stage: {name}')
        stage = config['evaluations'][name]
        split = stage['dataset_split']; per_task = stage['episodes_per_task']
        if (name, split, per_task) not in [('val40', 'train', 10), ('test200', 'test', 50)]:
            raise ValueError('val40 uses official train, 10/task; test200 uses official test, 50/task')
        if stage.get('episode_ids_json'):
            ids = json.loads(repo_path(stage['episode_ids_json']).read_text())
        else:
            ids = {task: list(range(per_task)) for task in tasks}
        if set(ids) != set(tasks):
            raise ValueError(f'Wrong task set in {name}')
        for task in tasks:
            values = ids[task]
            if len(values) != per_task or len(set(values)) != per_task or any(type(i) is not int for i in values):
                raise ValueError(f'Wrong/duplicate episode IDs in {name}/{task}')
            meta_path = REPO / 'robomme_sim/robomme/env_metadata' / split / f'record_dataset_{task}_metadata.json'
            meta = json.loads(meta_path.read_text()); records = meta['records']
            official_ids = {r['episode'] for r in records if r.get('task', meta.get('env_id')) == task}
            if meta['record_count'] != len(records) or len(official_ids) != len(records):
                raise ValueError(f'Invalid/duplicate official episode records: {meta_path}')
            if not set(values) <= official_ids:
                raise ValueError(f'Episode ID out of bounds: {name}/{task}')
            if name == 'test200' and (len(records) != 50 or set(values) != official_ids):
                raise ValueError(f'Expected all 50 official test episodes for {task}')
            metadata_hashes[str(meta_path.relative_to(REPO))] = sha(meta_path)
        stages[name] = {'dataset_split': split, 'episodes_per_task': per_task, 'episode_ids': ids}
    heldout = config['heldout_validation']
    train = read_episode_set(repo_path(heldout['train_episodes']))
    val = read_episode_set(repo_path(heldout['val_episodes']))
    if len(train) != 360 or len(val) != 40 or train & val:
        raise ValueError('Expected disjoint training360 / held-out40')
    if 'val40' in stages:
        mapped = {heldout['episode_bases'][task] + e for task, values in stages['val40']['episode_ids'].items() for e in values}
        if mapped != val:
            raise ValueError('val40 episode manifest differs from the actual training held-out split')
    files = ['robomme_sim/evaluation/valtest.py', 'robomme_sim/eval_success.py',
             'robomme_sim/robomme_env.py', 'robomme_sim/inproc_pool.py',
             'robomme_sim/policy.py', 'robomme_sim/batched_policy.py',
             'robomme_sim/subtask_memory_policy.py', 'robomme_sim/memcache_policy.py',
             'simplememvla/model/modeling_simplememvla.py', 'simplememvla/model/lora.py',
             'simplememvla/model/dit_action_head.py', 'simplememvla/model/memory.py',
             'simplememvla/data/messages.py', 'simplememvla/data/normalize.py',
             'scripts/eval_robomme_valtest.sh']
    identity = {'checkpoint': str(checkpoint), 'global_step': trainer.get('global_step'),
        'weights_size': weights.stat().st_size, 'weights_mtime_ns': weights.stat().st_mtime_ns,
        'checkpoint_config_sha256': sha(checkpoint / 'config.json'),
        'checkpoint_stats_sha256': sha(checkpoint / 'stats.json'),
        'resolved_config': config, 'stages': stages, 'official_metadata_sha256': metadata_hashes,
        'train_split_sha256': sha(repo_path(heldout['train_episodes'])),
        'val_split_sha256': sha(repo_path(heldout['val_episodes'])),
        'source_sha256': {name: sha(REPO / name) for name in files}}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return config, checkpoint, model, stages, identity, identity_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True, help='Physical GPU index; one evaluator per GPU')
    parser.add_argument('--output', type=Path, help='Default: unique directory derived from checkpoint')
    parser.add_argument('--splits', nargs='+', choices=['val40', 'test200'])
    parser.add_argument('--dry-run', action='store_true', help='CPU-only preflight; no evaluation or output creation')
    parser.add_argument('--resume', action='store_true', help='Reuse validated completed splits in --output')
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config, checkpoint, model, stages, identity, identity_hash = prepare(config_path, args.splits)
    if args.resume and not args.output:
        parser.error('--resume requires the previous --output directory')
    tag = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = repo_path(args.output) if args.output else repo_path(config['output_root']) / f'{checkpoint.parent.name}__{checkpoint.name}__{tag}'
    plan = {'checkpoint': str(checkpoint), 'global_step': identity['global_step'],
        'gpu': args.gpu, 'use_memory': bool(model.get('use_memory')), 'output': str(output),
        'evaluations': {name: {'dataset_split': s['dataset_split'],
            'episodes': sum(map(len, s['episode_ids'].values())), 'episode_ids': s['episode_ids']}
            for name, s in stages.items()}}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    lock = open(f'/dev/shm/simplememvla_eval_gpu{args.gpu}.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Refuse existing output: {output}; use --resume')
    if args.resume:
        previous = json.loads((output / 'manifest.json').read_text())
        if previous['identity_sha256'] != identity_hash:
            raise ValueError('Resume requires unchanged checkpoint, config, code and episode manifests')
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(output / 'manifest.json', {'created_at': now(), 'gpu': args.gpu,
            'identity_sha256': identity_hash, **identity})
        (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copy2(config_path, output / 'input_config.yaml')
        for relative in identity['source_sha256']:
            target = output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO / relative, target)
    status = {'state': 'running', 'pid': os.getpid(), 'gpu': args.gpu, 'started_at': now(), 'stage': None}
    summary = {'checkpoint': str(checkpoint), 'global_step': identity['global_step'],
               'use_memory': bool(model.get('use_memory')), 'evaluations': {}}
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    try:
        for name, stage in stages.items():
            status.update(stage=name, updated_at=now()); write_json(output / 'job_status.json', status)
            directory = output / name; directory.mkdir(exist_ok=True)
            protocol = {**config['protocol'], 'tasks': config['tasks'],
                'dataset_split': stage['dataset_split'], 'episodes_per_task': stage['episodes_per_task']}
            (directory / 'protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
            write_json(directory / 'episode_ids.json', stage['episode_ids'])
            result_path = directory / 'results.json'; done = directory / 'stage_status.json'
            complete = args.resume and done.exists() and json.loads(done.read_text()).get('state') == 'complete'
            if not complete:
                # Keep failed-attempt evidence while requiring a newly written result.
                for filename in ['results.json', 'eval_args.json']:
                    previous = directory / filename
                    if previous.exists():
                        previous.rename(previous.with_name(previous.stem + '.previous.' + uuid.uuid4().hex[:8] + '.json'))
                command = [sys.executable, '-m', 'robomme_sim.eval_success', '--config', str(directory / 'protocol.yaml'),
                    '--pretrained_checkpoint', str(checkpoint), '--episode_ids_json', str(directory / 'episode_ids.json'),
                    '--num_gpus', '1', '--log_file', str(directory / 'eval.log')]
                if protocol['video_max_per_task'] > 0:
                    command += ['--video_dir', str(directory / 'videos')]
                write_json(done, {'state': 'running', 'started_at': now(), 'command': command})
                with open(directory / 'launcher.log', 'a', buffering=1) as log:
                    process = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f'{name} exited {process.returncode}; see {directory / "launcher.log"}')
            validate_eval_args(directory, checkpoint, protocol)
            result = validate_result(result_path, stage['episode_ids'])
            write_json(done, {'state': 'complete', 'validated_at': now(), **result})
            summary['evaluations'][name] = result
            write_json(output / 'summary.json', summary)
            print(f'{name}: {result["successes"]}/{result["episodes"]} = {result["success_rate"]:.2%}', flush=True)
        status.update(state='complete', ended_at=now()); write_json(output / 'job_status.json', status)
        report = ['# RoboMME 闭环成功率', '', f'Checkpoint: `{checkpoint}`', '',
                  '| 数据集 | 成功/总数 | 成功率 |', '|---|---:|---:|']
        for name, result in summary['evaluations'].items():
            report.append(f'| {name} | {result["successes"]}/{result["episodes"]} | {result["success_rate"]:.2%} |')
        report += ['', 'val40：官方 train 中留出的内部验证集；test200：四任务官方 test 全部各50条。',
                   '闭环条件使用模型生成的 sub-task；E2 memory 仅使用当前 episode 已观测帧。', '']
        (output / 'REPORT_zh.md').write_text('\n'.join(report))
    except BaseException as error:
        status.update(state='failed', error=str(error), ended_at=now())
        write_json(output / 'job_status.json', status)
        traceback.print_exc(); raise


if __name__ == '__main__':
    main()
