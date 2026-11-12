




import os
import sys
import argparse
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_dataset4LLM.generate_dataset import (
    index_to_token_sequence,
    get_all_tokens,
)

def setup_logging(log_file=None):
    log_format = '%(asctime)s %(levelname)-8s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)

def load_model_for_inference(model_path, device="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", model_path)

        logger.info(f"加载基础模型: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info(f"加载完整模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer

def build_code_to_entity_map(entity_codes_path):
    with open(entity_codes_path) as f:
        entity_codes = json.load(f)

    code_to_entity, entity_to_code = {}, {}
    for eid_str, info in entity_codes.items():
        eid = int(eid_str)
        token_seq = info['token_sequence']
        code_to_entity[token_seq] = eid
        entity_to_code[eid] = token_seq

    return code_to_entity, entity_to_code

def prepare_entity_codes_gpu(entity_to_code, tokenizer, device, max_code_len=None):
    
    entity_ids = []
    code_token_ids = []
    code_lengths = []
    
    for eid, code in entity_to_code.items():
        tokens = tokenizer.encode(code, add_special_tokens=False)
        if tokens:
            entity_ids.append(eid)
            code_token_ids.append(tokens)
            code_lengths.append(len(tokens))
    
    if max_code_len is None:
        max_code_len = max(code_lengths) if code_lengths else 1
    
    
    padded_codes = []
    for tokens in code_token_ids:
        padded = tokens + [tokenizer.pad_token_id] * (max_code_len - len(tokens))
        padded_codes.append(padded)
    
    
    code_tensor = torch.tensor(padded_codes, dtype=torch.long, device=device)
    lengths_tensor = torch.tensor(code_lengths, dtype=torch.long, device=device)
    
    
    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
    
    
    length_to_indices = {}
    unique_lengths = torch.unique(lengths_tensor).cpu().numpy()
    for length in unique_lengths:
        length = int(length)
        mask = lengths_tensor == length
        indices = torch.where(mask)[0]  
        length_to_indices[length] = indices
    
    logger.info(f"预处理 {len(entity_ids)} 个实体码字，最大长度: {max_code_len}，长度组数: {len(length_to_indices)}")
    
    return entity_ids, code_tensor, lengths_tensor, entity_id_to_idx, length_to_indices

def compute_batch_scores_vectorized(batch_logits, seq_lengths, code_tensor, lengths_tensor,
                                    length_to_indices, device):
    
    batch_size = batch_logits.shape[0]
    num_entities = code_tensor.shape[0]
    dtype = batch_logits.dtype  
    batch_scores = torch.full((batch_size, num_entities), -float('inf'), device=device, dtype=dtype)
    
    
    lengths = list(length_to_indices.keys())
    
    for sample_idx in range(batch_size):
        seq_len = seq_lengths[sample_idx].item()
        for L in lengths:
            if L <= seq_len:
                indices = length_to_indices[L]                 
                codes = code_tensor[indices, :L]               
                start_pos = seq_len - L
                logits_block = batch_logits[sample_idx, start_pos:seq_len, :]  
                log_probs = F.log_softmax(logits_block, dim=-1)                
                
                
                idx_range = torch.arange(L, device=device).view(-1, 1)          
                token_log_probs = log_probs[idx_range, codes.T]                
                scores = token_log_probs.sum(dim=0)                            
                
                batch_scores[sample_idx, indices] = scores  
    
    return batch_scores

def evaluate_with_logits_topk(model, tokenizer, prompts, meta, entity_to_code,
                              device, batch_size=16, bad_case_path=None, good_case_path=None):
    
    logs, bad_cases, good_cases = [], [], []
    
    
    entity_ids, code_tensor, lengths_tensor, entity_id_to_idx, length_to_indices = \
        prepare_entity_codes_gpu(entity_to_code, tokenizer, device)
    
    num_entities = len(entity_ids)
    logger.info(f"开始极致优化评估...")
    logger.info(f"  样本数: {len(prompts)}")
    logger.info(f"  实体数: {num_entities}")
    logger.info(f"  批大小: {batch_size}")
    
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    start_time = time.time()
    
    
    model_dtype = next(model.parameters()).dtype
    use_amp = (device == 'cuda') and (model_dtype in (torch.float16, torch.bfloat16))
    amp_dtype = model_dtype if use_amp else torch.float32

    for batch_idx, i in enumerate(tqdm(range(0, len(prompts), batch_size), desc="评估中", total=total_batches)):
        batch_prompts = prompts[i:i + batch_size]
        batch_meta = meta[i:i + batch_size]
        batch_size_actual = len(batch_prompts)
        
        with torch.inference_mode():
            
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                prompt_encodings = tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(device)
                
                prompt_logits = model(**prompt_encodings).logits  
            
            
            seq_lengths = prompt_encodings.attention_mask.sum(dim=1)  
            
            
            batch_scores = compute_batch_scores_vectorized(
                prompt_logits, seq_lengths, code_tensor, lengths_tensor,
                length_to_indices, device
            )  
            
            
            batch_rankings = torch.full((batch_size_actual,), num_entities, dtype=torch.long, device=device)
            top1_indices = torch.empty(batch_size_actual, dtype=torch.long, device=device)
            top1_scores = torch.empty(batch_size_actual, dtype=torch.float, device=device)
            
            for j in range(batch_size_actual):
                target_eid = batch_meta[j]['answer_entity_id']
                target_idx = entity_id_to_idx[target_eid]
                target_score = batch_scores[j, target_idx]
                
                
                rank = (batch_scores[j] > target_score).sum().item() + 1
                batch_rankings[j] = rank
                
                
                top1_score, top1_idx = torch.max(batch_scores[j], dim=0)
                top1_indices[j] = top1_idx
                top1_scores[j] = top1_score
                
                
                if bad_case_path and rank > 10:
                    predicted_top1_eid = entity_ids[top1_idx.item()]
                    bad_cases.append({
                        'test_sample_index': i + j,
                        'prompt': batch_prompts[j],
                        'correct_answer_entity_id': target_eid,
                        'correct_answer_code': entity_to_code.get(target_eid, "N/A"),
                        'predicted_rank': rank,
                        'predicted_top1_entity_id': predicted_top1_eid,
                        'predicted_top1_code': entity_to_code.get(predicted_top1_eid, "N/A"),
                        'predicted_top1_score': float(top1_score.item())
                    })

                
                if good_case_path and rank <= 10:
                    predicted_top1_eid = entity_ids[top1_idx.item()]

                    
                    top10_scores, top10_indices = torch.topk(batch_scores[j], min(10, num_entities))
                    top10_entities = []
                    for k in range(len(top10_indices)):
                        idx = top10_indices[k].item()
                        score = top10_scores[k].item()
                        eid = entity_ids[idx]
                        top10_entities.append({
                            'entity_id': eid,
                            'fsq_code': entity_to_code.get(eid, "N/A"),
                            'score': float(score)
                        })

                    good_cases.append({
                        'test_sample_index': i + j,
                        'prompt': batch_prompts[j],
                        'correct_answer_entity_id': target_eid,
                        'correct_answer_code': entity_to_code.get(target_eid, "N/A"),
                        'predicted_rank': rank,
                        'predicted_top1_entity_id': predicted_top1_eid,
                        'predicted_top1_code': entity_to_code.get(predicted_top1_eid, "N/A"),
                        'predicted_top1_score': float(top1_score.item()),
                        'top10_predictions': top10_entities
                    })
        
        
        batch_rankings_cpu = batch_rankings.cpu().numpy()
        for j in range(batch_size_actual):
            ranking = batch_rankings_cpu[j]
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'Hits@1': 1.0 if ranking <= 1 else 0.0,
                'Hits@3': 1.0 if ranking <= 3 else 0.0,
                'Hits@10': 1.0 if ranking <= 10 else 0.0,
            })
        
        

    elapsed_time = time.time() - start_time
    speed = len(prompts) / elapsed_time
    logger.info(f"评估耗时: {elapsed_time:.2f}s，平均速度: {speed:.2f} samples/s")

    if bad_case_path and bad_cases:
        with open(bad_case_path, 'w', encoding='utf-8') as f:
            for case in bad_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        logger.info(f"Bad case analysis 已保存到: {bad_case_path}")

    if good_case_path and good_cases:
        with open(good_case_path, 'w', encoding='utf-8') as f:
            for case in good_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        logger.info(f"Good case analysis 已保存到: {good_case_path}")

    metrics = {key: sum(log[key] for log in logs) / len(logs) for key in logs[0].keys()}
    return metrics

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--test_data_path", type=str, required=True, help="")
    parser.add_argument("--entity_codes_path", type=str, required=True, help="")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", help="")
    parser.add_argument("--bad_case_path", type=str, default=None, help="")
    parser.add_argument("--good_case_path", type=str, default=None, help="")
    parser.add_argument("--log_file", type=str, default="evaluation.log", help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--device", type=str, default="auto", help="")
    args = parser.parse_args()

    
    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, args.log_file)
    setup_logging(log_file_path)

    bad_case_file_path = args.bad_case_path
    if bad_case_file_path is None:
        base_name = os.path.splitext(os.path.basename(args.output_path))[0]
        bad_case_file_path = os.path.join(output_dir, f"{base_name}_bad_cases.jsonl")

    good_case_file_path = args.good_case_path
    if good_case_file_path is None:
        base_name = os.path.splitext(os.path.basename(args.output_path))[0]
        good_case_file_path = os.path.join(output_dir, f"{base_name}_good_cases.jsonl")

    logger.info("="*60)
    logger.info("MMKGC 预测与评估（极致优化版本）")
    logger.info("="*60)
    logger.info(f"评估参数: {vars(args)}")

    
    logger.info("加载模型和 tokenizer...")
    model, tokenizer = load_model_for_inference(args.model_path, args.device)
    
    logger.info("加载测试数据和实体码字...")
    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    
    
    
    if isinstance(test_data, dict) and 'prompts' in test_data:
        
        prompts = test_data['prompts']
        meta = test_data.get('meta', [{}] * len(prompts))
    else:
        
        prompts = [item['prompt'] for item in test_data]
        meta = [item['meta'] for item in test_data]
    
    _, entity_to_code = build_code_to_entity_map(args.entity_codes_path)

    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    
    logger.info("开始基于 logits 的评估...")
    metrics = evaluate_with_logits_topk(
        model, tokenizer, prompts, meta, entity_to_code,
        device, args.batch_size, bad_case_file_path, good_case_file_path
    )

    
    logger.info("="*60)
    logger.info("评估完成!")
    logger.info("="*60)
    logger.info(f"  MRR: {metrics['MRR']:.4f}")
    logger.info(f"  MR: {metrics['MR']:.2f}")
    logger.info(f"  Hits@1: {metrics['Hits@1']:.4f}")
    logger.info(f"  Hits@3: {metrics['Hits@3']:.4f}")
    logger.info(f"  Hits@10: {metrics['Hits@10']:.4f}")

    with open(args.output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"评估结果已保存到: {args.output_path}")

if __name__ == "__main__":
    main()
