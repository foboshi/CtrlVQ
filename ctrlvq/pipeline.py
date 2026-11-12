
import contextlib
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .configuration import CORE, STAGES, artifacts, load, paths, requirements


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(CORE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CORE.rglob('*.py'))}


def fingerprint(cfg):
    cfg = dict(cfg)
    cfg['world_size'] = len(cfg.pop('gpu_ids'))
    return digest({'config': cfg, 'core': source_hashes()})


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def command(cfg, stage, config_path, resume_checkpoint=None):
    cmd = [sys.executable]
    if stage in ('quantize', 'stage1', 'stage2'):
        cmd += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc-per-node=' + str(len(cfg['gpu_ids'])), '--module', 'ctrlvq.entry']
    else:
        cmd += ['-m', 'ctrlvq.entry']
    cmd += [stage, '--config', str(config_path)]
    if resume_checkpoint:
        cmd += ['--resume-checkpoint', str(resume_checkpoint)]
    return cmd


def show_plan(cfg, config_path, stages):
    for stage in stages:
        print(f'[{stage}] {shlex.join(command(cfg, stage, config_path))}')
    print('\nCUDA_VISIBLE_DEVICES=' + ','.join(map(str, cfg['gpu_ids'])))
    print('Outputs: ' + cfg['output_dir'])
    print('Evaluation: original test_metrics_fast.py scoring (unfiltered)')


def checkpoint(directory):
    found = [p for p in directory.glob('checkpoint-*')
             if p.name.split('-')[-1].isdigit() and (p / 'training_state.pt').exists()]
    return max(found, key=lambda p: int(p.name.split('-')[-1])) if found else None


@contextlib.contextmanager
def lock(root):
    target = root / '.run.lock'
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Run lock exists: {target}. Check its PID before removing a stale lock.') from None
    with os.fdopen(fd, 'w') as f:
        json.dump({'pid': os.getpid(), 'created': time.time()}, f)
    try:
        yield
    finally:
        target.unlink(missing_ok=True)


def verify_artifacts(cfg, stage, log):
    missing = [str(p) for p in artifacts(cfg, stage) if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError('Missing output artifacts: ' + ', '.join(missing))
    if stage in ('stage1', 'stage2'):
        text = log.read_text(errors='replace')
        if '训练过程中发生错误:' in text:
            raise RuntimeError('The native trainer logged an error, even though it may have exited zero')
        if stage == 'stage1' and not list((paths(cfg)['stage1'] / 'final').glob('*.safetensors')):
            raise RuntimeError('Stage1 final model weights are missing')
    if stage == 'evaluate':
        values = json.loads(artifacts(cfg, stage)[0].read_text())
        for key in ('MRR', 'MR', 'Hits@1', 'Hits@3', 'Hits@10'):
            if key not in values or not math.isfinite(values[key]):
                raise RuntimeError('Invalid evaluation metric: ' + key)


def run(cfg, stages, resume=False):
    p = paths(cfg)
    p['root'].mkdir(parents=True, exist_ok=True)
    with lock(p['root']):
        p['metadata'].mkdir(exist_ok=True)
        p['logs'].mkdir(exist_ok=True)
        identity = fingerprint(cfg)
        manifest = p['metadata'] / 'run.json'
        previous = json.loads(manifest.read_text()) if manifest.exists() else None
        if previous and previous['fingerprint'] != identity:
            raise ValueError('Configuration or core source changed. Use a new output_dir.')
        if not previous:
            existing = [x for name in STAGES for x in p[name].glob('*')]
            if existing:
                raise ValueError('Unmanaged artifacts already exist in output_dir; choose a fresh directory')
        write(manifest, {'project': 'CtrlVQ', 'fingerprint': identity, 'config': cfg,
                         'core_sha256': source_hashes(), 'world_size': len(cfg['gpu_ids']),
                         'protocol': 'original_training_and_scoring', 'filtered': False})
        resolved = p['metadata'] / 'resolved_config.json'
        write(resolved, cfg)
        state = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'completed': []}
        status = p['metadata'] / 'status.json'
        write(status, state)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, cfg['gpu_ids'])), PYTHONHASHSEED=str(cfg['seed']),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
        
        package_parent = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = package_parent + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        child = None
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        try:
            for stage in stages:
                marker = p['metadata'] / (stage + '_complete.json')
                if marker.exists():
                    if not all(x.is_file() and x.stat().st_size for x in artifacts(cfg, stage)):
                        raise RuntimeError(f'{stage} completion marker has missing artifacts')
                    print(f'[{stage}] Already complete', flush=True)
                    state['completed'].append(stage)
                    continue
                missing = [str(x) for x in requirements(cfg, stage) if not x.is_file()]
                if missing:
                    raise FileNotFoundError(f'{stage} requires: ' + ', '.join(missing))
                target = p[stage]
                saved = checkpoint(target) if resume and stage in ('stage1', 'stage2') else None
                if target.exists() and any(target.iterdir()) and saved is None:
                    raise RuntimeError(f'Incomplete {stage} output exists; use --resume with a saved native checkpoint or a new output_dir')
                target.mkdir(parents=True, exist_ok=True)
                cmd = command(cfg, stage, resolved, saved)
                log = p['logs'] / (stage + f'-{time.time_ns()}.log')
                state.update(phase=stage, command=cmd, log=str(log))
                write(status, state)
                print(f'[{stage}] Running; log: {log}', flush=True)
                with log.open('w') as stream:
                    child = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    write(status, state)
                    code = child.wait()
                child = None
                if code:
                    raise RuntimeError(f'{stage} exited {code}; see {log}')
                verify_artifacts(cfg, stage, log)
                write(marker, {'fingerprint': identity, 'command': cmd, 'log': str(log), 'finished': time.time()})
                state['completed'].append(stage)
                write(status, state)
            state.update(state='complete', phase=None, finished=time.time())
        except KeyboardInterrupt:
            state.update(state='paused', paused_at=time.time())
            raise
        except Exception as exc:
            state.update(state='failed', error=str(exc), finished=time.time())
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            signal.signal(signal.SIGTERM, old_handler)
            write(status, state)
