"""Run reusable RoboMME action-error and success evaluation stages."""

import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import yaml

REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_success(path, episode_ids):
    result = json.loads(path.read_text())
    assert result['n_excluded_na'] == 0, result
    assert set(result['tasks']) == set(episode_ids), result
    for task, ids in episode_ids.items():
        row = result['tasks'][task]
        assert row['n'] == len(ids) and not row['oom'], (task, row)
        assert row['success_rate'] is not None, (task, row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--experiment', choices=['E1', 'E2'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Override configured checkpoint')
    parser.add_argument('--expected-step', type=int, help='Override configured step')
    parser.add_argument('--stages', choices=['all', 'openloop', 'success', 'report'], default='all')
    parser.add_argument('--resume', action='store_true', help='Reuse validated complete stages; retry incomplete stages')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    checkpoint = (args.checkpoint or Path(config['experiments'][args.experiment]['checkpoint'])).resolve()
    step = args.expected_step if args.expected_step is not None else config['expected_step']
    model_config = json.loads((checkpoint/'config.json').read_text())
    trainer_state = json.loads((checkpoint/'trainer_state.json').read_text())
    assert trainer_state['global_step'] == step, trainer_state['global_step']
    assert model_config['current_frame_only'] and model_config['condition_source'] == 'subtask'
    assert model_config['use_memory'] == (args.experiment == 'E2')
    assert model_config['action_horizon'] == 30
    weights = checkpoint/'model.safetensors'
    assert weights.is_file() and weights.stat().st_size > 1_000_000_000
    episodes_file = Path(config['episodes_json'])
    if not episodes_file.is_absolute():
        episodes_file = REPO/episodes_file
    episode_ids = json.loads(episodes_file.read_text())
    protocol = config['protocol']
    assert protocol['execute_horizon'] == 16 and protocol['batch_size'] == 4
    assert protocol['stride'] == 16 and protocol['expected_samples'] == 872
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Use a fresh output directory or --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    lock = open(f'/dev/shm/simplememvla_eval_gpu{args.gpu}.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert old['checkpoint'] == str(checkpoint) and old['step'] == step
        assert old['config_sha256'] == digest(args.config), 'Resume requires identical config'
        assert old['experiment'] == args.experiment
    else:
        snapshot = output/'source_snapshot'
        files = list((REPO/'robomme_sim/evaluation').glob('*.py')) + [
            REPO/'robomme_sim/eval_success.py', REPO/'robomme_sim/policy.py',
            REPO/'robomme_sim/batched_policy.py', REPO/'robomme_sim/subtask_memory_policy.py',
            REPO/'simplememvla/data/dataset.py', REPO/'simplememvla/data/collator.py',
            REPO/'simplememvla/data/normalize.py', REPO/'simplememvla/model/modeling_simplememvla.py',
            REPO/'simplememvla/model/dit_action_head.py', REPO/'simplememvla/model/memory.py',
            REPO/'simplememvla/model/lora.py', REPO/'simplememvla/model/configuration_simplememvla.py',
            REPO/'robomme_sim/robomme_env.py',
            REPO/'scripts/eval_posttrain.sh',
            REPO/'third_party/ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf',
        ]
        hashes = {}
        for source in files:
            relative = source.relative_to(REPO)
            dest = snapshot/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            hashes[str(relative)] = digest(source)
        for name in ['config.json', 'trainer_state.json', 'stats.json', 'train_config.yaml']:
            source = checkpoint/name
            if source.exists():
                dest = output/'checkpoint_metadata'/name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(source, dest)
        shutil.copy2(args.config, output/'run_config.yaml')
        shutil.copy2(episodes_file, output/'train_episode_ids.json')
        versions = {}
        for package in ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'sapien', 'flash-attn']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(output/'environment.json', {
            'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'], text=True),
        })
        (output/'git_state.txt').write_text(
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True)
            + subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True))
        write_json(manifest_path, {
            'experiment': args.experiment, 'step': step, 'checkpoint': str(checkpoint),
            'created_at': now(), 'gpu': args.gpu, 'config_sha256': digest(args.config),
            'weights_size': weights.stat().st_size, 'weights_mtime_ns': weights.stat().st_mtime_ns,
            'model_config_sha256': digest(checkpoint/'config.json'),
            'stats_sha256': digest(checkpoint/'stats.json'), 'source_sha256': hashes,
            'baseline_errors': config['baseline_errors'], 'baseline_success': config['baseline_success'],
            'protocol': protocol,
        })
    (output/'protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    closed_protocol = {k: protocol[k] for k in [
        'dataset_split', 'group_size', 'current_frame_only', 'fail_on_env_error',
        'execute_horizon', 'max_steps', 'num_denoising_steps', 'eval_temperature',
        'max_subtask_tokens', 'compute_dtype', 'attn_implementation', 'pipelined',
        'video_max_per_task', 'worker_timeout',
    ]}
    closed_protocol.update(tasks=list(episode_ids), episodes_per_task=10)
    (output/'success_protocol.yaml').write_text(yaml.safe_dump(closed_protocol, sort_keys=False))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    status = {'experiment': args.experiment, 'step': step, 'gpu': args.gpu, 'pid': os.getpid(),
              'started_at': now(), 'state': 'running', 'stage': None}
    write_json(output/'job_status.json', status)

    def run_stage(stage, command):
        status.update(stage=stage, updated_at=now())
        write_json(output/'job_status.json', status)
        print(f'[{now()}] {stage}: {command}', flush=True)
        with open(output/(stage+'.log'), 'a', buffering=1) as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed with exit {result.returncode}; see {output/(stage+".log")}')

    try:
        if args.stages in ['all', 'openloop']:
            done = output/'openloop/status.json'
            complete = done.exists() and json.loads(done.read_text()).get('state') == 'complete'
            if not complete:
                run_stage('openloop', [sys.executable, '-m', 'robomme_sim.evaluation.action_errors',
                    '--experiment', args.experiment, '--checkpoint', str(checkpoint),
                    '--output', str(output/'openloop'), '--episodes-json', str(output/'train_episode_ids.json'),
                    '--protocol', str(output/'protocol.yaml')])
            run_stage('action_report', [sys.executable, '-m', 'robomme_sim.evaluation.report',
                '--output', str(output), '--kind', 'actions'])
        if args.stages in ['all', 'success']:
            (output/'success').mkdir(exist_ok=True)
            result_path = output/'success/results.json'
            if not result_path.exists():
                run_stage('success', [sys.executable, '-m', 'robomme_sim.eval_success',
                    '--config', str(output/'success_protocol.yaml'), '--pretrained_checkpoint', str(checkpoint),
                    '--num_gpus', '1', '--log_file', str(output/'success/eval.log'),
                    '--video_dir', str(output/'success/videos'),
                    '--episode_ids_json', str(output/'train_episode_ids.json')])
            validate_success(result_path, episode_ids)
        run_stage('comparison', [sys.executable, '-m', 'robomme_sim.evaluation.report',
            '--output', str(output), '--kind', 'comparison'])
        status.update(state='complete', stage=args.stages, ended_at=now())
        write_json(output/'job_status.json', status)
        print('EVALUATION_COMPLETE', args.experiment, output, flush=True)
    except BaseException as error:
        status.update(state='failed', error=str(error), ended_at=now())
        write_json(output/'job_status.json', status)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
