
import argparse
import os
import random
import runpy
import sys

from .configuration import CORE, SCRIPTS, STAGES, arguments, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-checkpoint')
    args = parser.parse_args()
    cfg = load(args.config)
    sys.path.insert(0, str(CORE))
    
    from configs import default_config
    default_config.DATASET_CONFIG.clear()
    default_config.DATASET_CONFIG[cfg['dataset']] = dict(cfg['data'])
    default_config.FUSION_CONFIG.update(cfg.get('fusion', {}))
    import numpy as np
    import torch
    seed = cfg['seed'] + int(os.environ.get('RANK', '0'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    script = CORE / SCRIPTS[args.stage]
    sys.argv = [str(script)] + arguments(cfg, args.stage, args.resume_checkpoint)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
