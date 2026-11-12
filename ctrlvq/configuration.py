
import copy
import json
import os
import re
from pathlib import Path

STAGES = ('quantize', 'data', 'stage1', 'stage2', 'evaluate')
CORE = Path(__file__).resolve().parent / 'core'
SCRIPTS = {
    'quantize': 'fsq_quantization/train_fsq.py',
    'data': 'gen_dataset4LLM/generate_dataset.py',
    'stage1': 'finetuning_token.py',
    'stage2': 'finetuning_LoRA.py',
    'evaluate': 'test_metrics_fast.py',
}
OPTIONS = {
    'quantizer': {'levels', 'num_codebooks', 'fused_emb_dim', 'latent_dim', 'fusion_method',
                  'batch_size', 'lr', 'num_epochs', 'recon_weight', 'diversity_weight',
                  'contrastive_weight', 'use_adapter', 'use_gumbel', 'temperature'},
    'stage1': {'batch_size', 'grad_accum', 'lr', 'weight_decay', 'warmup_steps', 'max_steps',
               'save_steps', 'eval_steps', 'early_stopping_patience', 'max_length',
               'no_amp', 'use_gradient_checkpointing'},
    'stage2': {'batch_size', 'grad_accum', 'lr', 'weight_decay', 'warmup_steps', 'max_steps',
               'save_steps', 'eval_steps', 'early_stopping_patience', 'max_length', 'last_k',
               'only_answer_loss', 'lora_r', 'lora_alpha', 'lora_dropout', 'no_amp',
               'max_grad_norm', 'use_cosine_annealing', 'use_gradient_checkpointing'},
    'evaluation': {'checkpoint', 'batch_size', 'device'},
}


def load(path):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    required = {'project', 'dataset', 'seed', 'gpu_ids', 'output_dir', 'model_path', 'data',
                'quantizer', 'stage1', 'stage1_checkpoint', 'stage2', 'evaluation'}
    missing = required - cfg.keys()
    if missing:
        raise ValueError('Missing config keys: ' + ', '.join(sorted(missing)))
    if cfg['dataset'] not in ('MKG-W', 'MKG-Y', 'MMKB-DB15K'):
        raise ValueError('Unsupported dataset')
    if cfg['seed'] != 42:
        raise ValueError('This native protocol supports seed 42; original quantizer fixes that seed internally')
    if not isinstance(cfg['gpu_ids'], list) or not cfg['gpu_ids']:
        raise ValueError('gpu_ids must contain at least one physical GPU index')
    if any(type(i) is not int or i < 0 for i in cfg['gpu_ids']) or len(set(cfg['gpu_ids'])) != len(cfg['gpu_ids']):
        raise ValueError('gpu_ids must be distinct nonnegative integers')
    for section, allowed in OPTIONS.items():
        extra = cfg[section].keys() - allowed
        if extra:
            raise ValueError(f'Unknown {section} options: {sorted(extra)}')
        for key in ('batch_size', 'max_steps', 'save_steps', 'eval_steps', 'num_epochs', 'max_length'):
            if key in cfg[section] and (type(cfg[section][key]) is not int or cfg[section][key] <= 0):
                raise ValueError(f'{section}.{key} must be a positive integer')
    if not cfg['quantizer'].get('use_adapter', True):
        raise ValueError('Original CLI cannot disable adapter; this would require a separate ablation implementation')
    if not cfg['stage1'].get('use_gradient_checkpointing', True):
        raise ValueError('Original Stage1 CLI always enables gradient checkpointing')
    if cfg['stage2'].get('only_answer_loss', False):
        raise ValueError('This compatibility profile preserves original last_k supervision')
    for value in (cfg['stage1_checkpoint'], cfg['evaluation']['checkpoint']):
        if not re.fullmatch(r'final|best_model|checkpoint-\d+', value):
            raise ValueError('Checkpoint must be final, best_model or checkpoint-N')
    cfg = copy.deepcopy(cfg)
    for key in ('output_dir', 'model_path'):
        cfg[key] = resolve(cfg[key], path.parent)
    for key in ('data_path', 'text_emb_path', 'image_emb_path', 'ent_link_path', 'structural_emb_path'):
        if key not in cfg['data']:
            raise ValueError('Missing data path: ' + key)
        cfg['data'][key] = resolve(cfg['data'][key], path.parent)
    return cfg


def resolve(value, base):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if '$' in value:
        raise ValueError('Unresolved environment variable in path: ' + value)
    path = Path(value)
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def paths(cfg):
    root = Path(cfg['output_dir'])
    return {key: root / name for key, name in {
        'root': '.', 'quantize': 'quantizer', 'data': 'datasets', 'stage1': 'stage1',
        'stage2': 'stage2', 'evaluate': 'evaluation', 'logs': 'logs', 'metadata': 'metadata',
    }.items()}


def arguments(cfg, stage, resume_checkpoint=None):
    p = paths(cfg)
    if stage == 'quantize':
        values = dict(cfg['quantizer'], dataset=cfg['dataset'], save_dir=p['quantize'])
    elif stage == 'data':
        values = {'dataset': cfg['dataset'], 'data_path': cfg['data']['data_path'],
                  'fsq_dir': p['quantize'], 'output_dir': p['data']}
    elif stage == 'stage1':
        values = dict(cfg['stage1'], model_name=cfg['model_path'], model_source='huggingface',
                      data_path=p['data'] / 'stage1_token_data.json', tokens_path=p['data'] / 'new_tokens.json',
                      output_dir=p['stage1'])
    elif stage == 'stage2':
        values = dict(cfg['stage2'], stage1_model_path=p['stage1'] / cfg['stage1_checkpoint'],
                      train_data_path=p['data'] / 'stage2_train_data.json',
                      valid_data_path=p['data'] / 'stage2_valid_data.json', output_dir=p['stage2'])
    else:
        values = {'model_path': p['stage2'] / cfg['evaluation']['checkpoint'],
                  'test_data_path': p['data'] / 'test_prompts.json',
                  'entity_codes_path': p['data'] / 'entity_codes.json',
                  'output_path': p['evaluate'] / 'metrics.json',
                  'batch_size': cfg['evaluation']['batch_size'], 'device': cfg['evaluation']['device']}
    if resume_checkpoint:
        if stage not in ('stage1', 'stage2'):
            raise ValueError('Native resume supported only for Stage1 and Stage2')
        values['resume_from_checkpoint'] = str(resume_checkpoint)
    args = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        args.append('--' + key)
        if value is not True:
            args.extend(map(str, value if isinstance(value, list) else [value]))
    return args


def requirements(cfg, stage):
    p = paths(cfg)
    if stage == 'quantize':
        return [Path(cfg['data']['data_path']) / (s + '.txt') for s in ('train', 'valid', 'test')] + [
            Path(cfg['data'][key]) for key in ('text_emb_path', 'image_emb_path', 'ent_link_path', 'structural_emb_path')]
    if stage == 'data':
        return [p['quantize'] / name for name in ('entity_fsq_indices.npy', 'config.json', 'entity2id.json',
                'relation2id.json', 'id2entity.json', 'id2relation.json')] + [
                Path(cfg['data']['data_path']) / name for name in ('train.txt', 'valid.txt', 'test.txt', 'one_hop.json')]
    if stage == 'stage1':
        return [Path(cfg['model_path']) / 'config.json', Path(cfg['model_path']) / 'tokenizer_config.json',
                p['data'] / 'stage1_token_data.json', p['data'] / 'new_tokens.json']
    if stage == 'stage2':
        return [p['stage1'] / cfg['stage1_checkpoint'] / 'config.json',
                p['data'] / 'stage2_train_data.json', p['data'] / 'stage2_valid_data.json']
    return [p['stage2'] / cfg['evaluation']['checkpoint'] / 'adapter_config.json',
            p['data'] / 'test_prompts.json', p['data'] / 'entity_codes.json']


def artifacts(cfg, stage):
    p = paths(cfg)
    names = {'quantize': ['config.json', 'entity_fsq_indices.npy', 'entity2id.json'],
             'data': ['stage1_token_data.json', 'stage2_train_data.json', 'stage2_valid_data.json',
                      'test_prompts.json', 'new_tokens.json', 'entity_codes.json'],
             'stage1': ['final/config.json', 'final/tokenizer_config.json'],
             'stage2': ['final/adapter_config.json', 'final/adapter_model.safetensors'],
             'evaluate': ['metrics.json']}
    return [p[stage] / name for name in names[stage]]
