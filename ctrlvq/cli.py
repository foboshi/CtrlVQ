
import argparse
import json
import sys
from pathlib import Path

from .configuration import STAGES, load, paths
from .pipeline import run, show_plan


def main():
    parser = argparse.ArgumentParser(prog='ctrlvq', description='CtrlVQ 原始流程的独立整理版本')
    parser.add_argument('action', choices=['plan', 'run', 'status'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', *STAGES], default='all')
    parser.add_argument('--gpus', help='Physical GPU indices, e.g. 0 or 0,1')
    parser.add_argument('--resume', action='store_true', help='Resume a partial native Stage1/Stage2 checkpoint')
    args = parser.parse_args()
    try:
        cfg = load(args.config)
        if args.gpus:
            ids = [int(x) for x in args.gpus.split(',')]
            if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
                raise ValueError('--gpus requires distinct nonnegative integer indices')
            cfg['gpu_ids'] = ids
        stages = STAGES if args.stage == 'all' else (args.stage,)
        if args.action == 'plan':
            show_plan(cfg, args.config.resolve(), stages)
        elif args.action == 'status':
            target = paths(cfg)['metadata'] / 'status.json'
            print(target.read_text() if target.exists() else json.dumps({'state': 'not_started'}, indent=2))
        else:
            run(cfg, stages, args.resume)
    except KeyboardInterrupt:
        print('CtrlVQ paused; saved native checkpoints are retained.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'CtrlVQ: {exc}\n')


if __name__ == '__main__':
    main()
