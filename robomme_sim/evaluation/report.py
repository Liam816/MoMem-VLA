"""Validate paired predictions and write machine-readable and Chinese reports."""

import argparse
import csv
import datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .kinematics import PandaFK
from .metrics import summarize_metrics


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def export_actions(output, manifest):
    folder = output/'openloop'
    baseline = Path(manifest['baseline_errors'])/manifest['experiment']
    raw = dict(np.load(folder/'predictions.npz', allow_pickle=False))
    reference = dict(np.load(baseline/'predictions.npz', allow_pickle=False))
    meta = [json.loads(line) for line in (folder/'per_prediction.jsonl').read_text().splitlines() if line.strip()]
    assert len(meta) == len(raw['gt']) == manifest['protocol']['expected_samples']
    for key in ['episode', 'frame', 'valid']:
        assert np.array_equal(raw[key], reference[key]), f'1k/5k {key} mismatch'
    for key in ['gt', 'state', 'timesteps']:
        assert np.allclose(raw[key], reference[key], atol=2e-6), f'1k/5k {key} mismatch'
    assert all(np.isfinite(value).all() for value in raw.values()), 'Non-finite predictions'
    summary = load(folder/'openloop_summary.json')
    old_summary = load(baseline/'openloop_summary.json')
    assert summary['action_std'] == old_summary['action_std']
    assert summary['action_mean'] == old_summary['action_mean']
    fk = PandaFK()
    # A regression oracle uses the saved 1k arrays and the saved 1k report.
    # This checks the refactor without running another GPU evaluation.
    oracle = summarize_metrics(reference, fk, np.asarray(summary['action_mean']), np.asarray(summary['action_std']))
    for mode in ['teacher_forced', 'generated', 'hold_current_joints']:
        for key, value in oracle[mode]['first_16'].items():
            assert np.allclose(value, old_summary['overall'][mode]['first_16'][key], atol=1e-9), (mode, key)
    save(folder/'comparability_checks.json', {
        'same_872_episode_frame_anchors': True, 'same_raw_gt_state_valid_mask': True,
        'same_normalization': True, 'same_flow_timesteps': True,
        'refactored_metrics_reproduce_saved_1k': True, 'samples': len(meta),
    })
    gt_pose = fk(raw['gt'][..., :7])
    flat = []
    for mode, key in [('teacher_forced', 'tf'), ('generated', 'gen')]:
        error = raw[key] - raw['gt']
        pose = fk(raw[key][..., :7])
        tcp = np.linalg.norm(pose[..., :3, 3] - gt_pose[..., :3, 3], axis=-1) * 1000
        with gzip.open(folder/f'per_action_errors_{mode}.csv.gz', 'wt', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['dataset_episode', 'frame', 'horizon_1based', 'valid',
                *[f'joint{i+1}_error_rad' for i in range(7)], 'gripper_error', 'tcp_error_mm'])
            for i, item in enumerate(meta):
                for horizon in range(30):
                    writer.writerow([item['episode'], item['frame'], horizon+1,
                        int(raw['valid'][i, horizon]), *error[i, horizon].tolist(), float(tcp[i, horizon])])
                valid = raw['valid'][i, :16]
                per_joint = np.abs(error[i, :16, :7][valid]).mean(0)
                flat.append({'model': manifest['experiment'], 'step': manifest['step'], 'condition': mode,
                    'task': item['task'], 'dataset_episode': item['episode'], 'frame': item['frame'],
                    'control_step': item['elapsed_control_steps'], 'valid_horizon': item['valid_horizon'],
                    'gt_subtask': item['gt_subtask'], 'generated_subtask': item['generated_subtask'],
                    'same_condition_tokens': item['same_condition_tokens'],
                    'joint_mae_first16_rad': float(per_joint.mean()),
                    'joint_mae_first16_deg': float(np.rad2deg(per_joint.mean())),
                    'tcp_error_first16_mean_mm': float(tcp[i, :16][valid].mean()),
                    **{f'joint{j+1}_mae_first16_rad': float(per_joint[j]) for j in range(7)}})
        with open(folder/'per_prediction_summary.csv', 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def comparison(output, manifest):
    experiment = manifest['experiment']
    step = manifest['step']
    lines = [f'# {experiment}：step1000 与 step{step} test on train 对比', '',
        f"step{step} 断点：`{manifest['checkpoint']}`", '',
        '同一 train40：PickXtimes、VideoUnmaskSwap、VideoRepick、PatternLock 各 10 条。',
        '开环使用 GT 观测，每模型 872 个采样点；预测 30×8，统计前 16 步，排除末尾 padding。',
        'TCP 误差是预测/GT 绝对关节目标经 Panda URDF 正运动学后的距离（mm），不是机器人实际闭环轨迹误差。',
        'generated 为部署时生成 sub-task 条件；teacher_forced 为训练 GT sub-task 条件。', '']
    result = {'experiment': experiment, 'step': step, 'actions': None, 'success': None}
    current_path = output/'openloop/openloop_summary.json'
    if current_path.exists():
        assert (output/'openloop/comparability_checks.json').exists()
        old = load(Path(manifest['baseline_errors'])/experiment/'openloop_summary.json')
        new = load(current_path)
        result['actions'] = {}
        result['diagnostics'] = {
            'step1000': {'flow_velocity_mse': old['overall']['flow_velocity_mse'],
                         'exact_subtask_match_rate': old['exact_generated_subtask_match_rate']},
            f'step{step}': {'flow_velocity_mse': new['overall']['flow_velocity_mse'],
                           'exact_subtask_match_rate': new['exact_generated_subtask_match_rate']},
        }
        lines += ['| 条件 | 断点 | 关节 MAE(rad) | 关节 MAE(°) | TCP mean(mm) | TCP P95(mm) |',
                  '|---|---:|---:|---:|---:|---:|']
        for mode in ['generated', 'teacher_forced']:
            a = old['overall'][mode]['first_16']
            b = new['overall'][mode]['first_16']
            result['actions'][mode] = {'step1000': a, f'step{step}': b,
                'joint_mae_rad_reduction': a['joint_mae_rad']-b['joint_mae_rad'],
                'tcp_mean_mm_reduction': a['tcp_mean_mm']-b['tcp_mean_mm'],
                'joint_mae_relative_reduction': 1-b['joint_mae_rad']/a['joint_mae_rad'],
                'tcp_mean_relative_reduction': 1-b['tcp_mean_mm']/a['tcp_mean_mm']}
            for label, row in [(1000, a), (step, b)]:
                lines.append(f"| {mode} | {label} | {row['joint_mae_rad']:.6f} | {row['joint_mae_deg']:.3f} | {row['tcp_mean_mm']:.3f} | {row['tcp_p95_mm']:.3f} |")
        lines += ['', '误差变化在 comparison.json 中以 old − new 表示，正值代表改善。', '']
        lines += [f"GT 条件 flow velocity MSE：{old['overall']['flow_velocity_mse']:.6f} → {new['overall']['flow_velocity_mse']:.6f}。",
                  f"生成 sub-task 严格文本匹配率：{100*old['exact_generated_subtask_match_rate']:.2f}% → {100*new['exact_generated_subtask_match_rate']:.2f}%。严格字符串匹配不等同于语义正确率。", '']
        lines += ['| 任务 | generated 关节MAE(°)，1k → 当前 | generated TCP(mm)，1k → 当前 | 当前 GT条件 TCP(mm) |',
                  '|---|---:|---:|---:|']
        result['actions_per_task'] = {}
        for task, task_summary in new['per_task'].items():
            a = old['per_task'][task]['generated']['first_16']
            b = task_summary['generated']['first_16']
            gt = task_summary['teacher_forced']['first_16']
            result['actions_per_task'][task] = {'step1000_generated': a,
                f'step{step}_generated': b, f'step{step}_teacher_forced': gt}
            lines.append(f"| {task} | {a['joint_mae_deg']:.3f} → {b['joint_mae_deg']:.3f} | {a['tcp_mean_mm']:.3f} → {b['tcp_mean_mm']:.3f} | {gt['tcp_mean_mm']:.3f} |")
        lines.append('')
        draw_plot(output, old, new, step)
    current_success = output/'success/results.json'
    if current_success.exists():
        old = load(Path(manifest['baseline_success'])/experiment/'results.json')
        new = load(current_success)
        result['success'] = {'step1000': old, f'step{step}': new,
            'change_percentage_points': 100*(new['macro_avg']-old['macro_avg'])}
        lines += ['| 闭环任务 | step1000 | step'+str(step)+' | 变化(百分点) |', '|---|---:|---:|---:|']
        for task in new['tasks']:
            a, b = old['tasks'][task], new['tasks'][task]
            lines.append(f"| {task} | {100*a['success_rate']:.1f}% ({round(a['success_rate']*a['n'])}/{a['n']}) | {100*b['success_rate']:.1f}% ({round(b['success_rate']*b['n'])}/{b['n']}) | {100*(b['success_rate']-a['success_rate']):+.1f} |")
        lines += [f"| **总体** | **{100*old['macro_avg']:.1f}%** | **{100*new['macro_avg']:.1f}%** | **{100*(new['macro_avg']-old['macro_avg']):+.1f}** |", '']
    else:
        lines += ['闭环成功率：尚未完成。', '']
    lines += ['采样/随机数口径：batch=4；action seed=20000+batch_index；flow seed=90000+batch_index；闭环 torch/numpy seed=0。',
              '闭环协议：group=5、execute_horizon=16、max_steps=1300、ODE=10、temperature=1、sub-task≤64、bf16/FA2、单帧。',
              '每条逐动作误差见 openloop/per_action_errors_*.csv.gz；原始预测/GT/有效 mask 见 openloop/predictions.npz。',
              '本轮只评测训练集抽样，不代表官方 test 成功率。', '']
    save(output/'comparison.json', result)
    (output/'REPORT_zh.md').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def compare_run(root):
    """Refresh a two-model summary after either model finishes a stage."""
    with open(root/'report.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        models = {}
        for experiment in ['E1', 'E2']:
            path = root/experiment/'comparison.json'
            if path.exists():
                models[experiment] = load(path)
        complete = len(models) == 2 and all(row['actions'] and row['success'] for row in models.values())
        summary = {'state': 'complete' if complete else 'running',
                   'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'experiments': models}
        save(root/'comparison.json', summary)
        steps = sorted({value['step'] for value in models.values()})
        current_label = 'step'+str(steps[0]) if len(steps) == 1 else '当前断点'
        lines = [f'# E1/E2 {current_label} 与 step1000 test on train 对比', '',
                 '状态：'+('已完成。' if complete else '运行中，以下仅列已完成指标。'), '',
                 '相同 train40 episode；每模型 872 个 GT 观测锚点；预测 30×8、统计前 16 步；padding 不计。',
                 'TCP 为绝对关节命令经 Panda 正运动学后的误差，不是实际闭环轨迹偏差。', '',
                 f'| 模型/条件 | 1k 关节MAE(°) | {current_label} 关节MAE(°) | 1k TCP mean(mm) | {current_label} TCP mean(mm) | TCP改善(%) |',
                 '|---|---:|---:|---:|---:|---:|']
        for experiment, value in models.items():
            for mode, row in (value['actions'] or {}).items():
                old, new = row['step1000'], row[f"step{value['step']}"]
                lines.append(f"| {experiment}/{mode} | {old['joint_mae_deg']:.3f} | {new['joint_mae_deg']:.3f} | {old['tcp_mean_mm']:.3f} | {new['tcp_mean_mm']:.3f} | {100*row['tcp_mean_relative_reduction']:+.1f}% |")
        lines += ['', f'| 闭环成功率 | 1k | {current_label} | 变化(百分点) |', '|---|---:|---:|---:|']
        for experiment, value in models.items():
            row = value['success']
            if row:
                latest = row['step'+str(value['step'])]['macro_avg']
                lines.append(f"| {experiment} | {100*row['step1000']['macro_avg']:.1f}% | {100*latest:.1f}% | {row['change_percentage_points']:+.1f} |")
        lines += ['', 'generated：模型生成 sub-task；teacher_forced：GT sub-task。成功率使用模型生成条件。',
                  '正改善表示误差下降；成功率变化为当前断点 − 1k。每任务成功率及 P95 等见 E1/REPORT_zh.md、E2/REPORT_zh.md。',
                  '本轮只覆盖训练集抽样，不代表官方 test 指标。手动指令见 MANUAL_zh.md。', '']
        temporary = root/'REPORT_zh.md.tmp'
        temporary.write_text('\n'.join(lines))
        temporary.replace(root/'REPORT_zh.md')
        launch = root/'launch_manifest.json'
        if launch.exists():
            info = load(launch)
            info.update(state=summary['state'], updated_at=summary['updated_at'])
            save(launch, info)
            pointer = root.parent/'latest_step5000_eval.json'
            if pointer.exists() and load(pointer).get('root') == str(root):
                save(pointer, info)


def draw_plot(output, old, new, step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, data, color in [(1000, old, '#888888'), (step, new, '#187ba6')]:
        for mode, style in [('generated', '-'), ('teacher_forced', '--')]:
            row = data['overall'][mode]
            name = f'{label}: '+('generated' if mode == 'generated' else 'GT subtask')
            axes[0].plot(range(1, 31), np.rad2deg(row['horizon_joint_mae_rad']), style, color=color, label=name)
            axes[1].plot(range(1, 31), row['horizon_tcp_mean_mm'], style, color=color, label=name)
    for axis in axes:
        axis.axvspan(1, 16, color='#eef4e8', alpha=0.6)
        axis.axvline(16, color='#999999', linewidth=0.8)
        axis.set_xlabel('Predicted command in chunk')
        axis.grid(alpha=0.2)
    axes[0].set_ylabel('Joint MAE (degrees)')
    axes[1].set_ylabel('TCP command error (mm)')
    axes[0].legend(fontsize=8)
    figure.suptitle(output.name+': paired GT-observation predictions')
    figure.savefig(output/'action_comparison.png', dpi=160)
    figure.savefig(output/'action_comparison.svg')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['actions', 'comparison'], default='comparison')
    args = parser.parse_args()
    manifest = load(args.output/'manifest.json')
    # Reports run after GPU prediction; record the report revision actually used.
    source = Path(__file__).resolve()
    source_key = 'robomme_sim/evaluation/report.py'
    current_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest['source_sha256'].get(source_key) != current_hash:
        manifest.setdefault('report_source_revisions', []).append({
            'previous_sha256': manifest['source_sha256'].get(source_key),
            'sha256': current_hash, 'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        destination = args.output/'source_snapshot'/source_key
        shutil.copy2(source, destination)
        manifest['source_sha256'][source_key] = current_hash
        save(args.output/'manifest.json', manifest)
    if args.kind == 'actions':
        export_actions(args.output, manifest)
    comparison(args.output, manifest)
    if (args.output.parent/'launch_manifest.json').exists():
        compare_run(args.output.parent)


if __name__ == '__main__':
    main()
