

import os
import sys
import argparse
import json
import logging
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_embedding.load_embeddings import (
    build_entity_relation_ids,
    read_triples,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_relation_names(data_path):
    
    relation_names_path = os.path.join(data_path, "relation_names.json")
    wikidata_relations = {}
    if os.path.exists(relation_names_path):
        with open(relation_names_path, 'r', encoding='utf-8') as f:
            wikidata_relations = json.load(f)
        logger.info(f"已加载 {len(wikidata_relations)} 个关系名称映射")
    else:
        logger.warning(f"关系名称文件不存在: {relation_names_path}")
    return wikidata_relations


def find_duplicate_entities(entity_indices):
    
    from collections import defaultdict
    code_to_entities = defaultdict(list)
    for entity_id, fsq_idx in entity_indices.items():
        code_to_entities[fsq_idx].append(entity_id)

    
    duplicate_codes = {code: entities for code, entities in code_to_entities.items() if len(entities) > 1}

    
    entities_need_unique_eid = set()
    entities_with_eid_not = set()

    for code, entities in code_to_entities.items():
        if len(entities) > 1:
            
            for i, eid in enumerate(entities):
                if i == 0:
                    entities_with_eid_not.add(eid)  
                else:
                    entities_need_unique_eid.add(eid)
        else:
            
            entities_with_eid_not.add(entities[0])

    return duplicate_codes, entities_need_unique_eid, entities_with_eid_not


def index_to_token_sequence(index, levels, token_prefix="[MM", token_suffix="]", entity_id=None,
                           eid_prefix="[EID", eid_suffix="]",
                           entities_need_unique_eid=None, entities_with_eid_not=None,
                           eid_not_token="[EID_NOT]"):
    
    sub_indices = []
    remaining = int(index)
    for level in levels:
        sub_idx = remaining % level
        sub_indices.append(sub_idx)
        remaining = remaining // level

    tokens = []
    for dim_idx, sub_idx in enumerate(sub_indices):
        tokens.append(f"{token_prefix}{dim_idx}_{sub_idx}{token_suffix}")

    
    if entity_id is not None:
        if entities_need_unique_eid is not None and entity_id in entities_need_unique_eid:
            
            tokens.append(f"{eid_prefix}_{entity_id}{eid_suffix}")
        elif entities_with_eid_not is not None and entity_id in entities_with_eid_not:
            
            tokens.append(eid_not_token)
        else:
            
            tokens.append(eid_not_token)

    token_seq = "".join(tokens)
    return token_seq, sub_indices


def get_all_tokens(levels, num_entities, token_prefix="[MM", token_suffix="]",
                   eid_prefix="[EID", eid_suffix="]",
                   entities_need_unique_eid=None, entities_with_eid_not=None,
                   eid_not_token="[EID_NOT]"):
    
    all_tokens = []

    
    for dim_idx, level in enumerate(levels):
        for sub_idx in range(level):
            token = f"{token_prefix}{dim_idx}_{sub_idx}{token_suffix}"
            all_tokens.append(token)

    
    if entities_with_eid_not:
        all_tokens.append(eid_not_token)

    
    if entities_need_unique_eid is not None:
        for entity_id in entities_need_unique_eid:
            token = f"{eid_prefix}_{entity_id}{eid_suffix}"
            all_tokens.append(token)

    token2id = {t: i for i, t in enumerate(all_tokens)}
    return all_tokens, token2id


def load_neighborhood(json_path):
    
    if not os.path.exists(json_path):
        logger.warning(f"邻居文件不存在: {json_path}")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        one_hop = json.load(f)
    one_hop = {int(k): v for k, v in one_hop.items()}
    logger.info(f"已加载一阶邻居信息: {len(one_hop)} 个实体")
    return one_hop


def load_entity_names(data_path):
    
    ent_links_path = os.path.join(data_path, "ent_links")
    url2name = {}
    if os.path.exists(ent_links_path):
        with open(ent_links_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    dbpedia_url = parts[0]
                    wikidata_url = parts[1]
                    name = dbpedia_url.split('/')[-1]
                    name = name.replace('_', ' ')
                    url2name[wikidata_url] = name
        logger.info(f"已加载 {len(url2name)} 个实体名称映射")
    return url2name


def get_relation_name(wikidata_url, wikidata_relations):
    
    if wikidata_url in wikidata_relations:
        return wikidata_relations[wikidata_url]
    
    return wikidata_url


def get_entity_name(wikidata_url, url2name):
    
    if url2name and wikidata_url in url2name:
        return url2name[wikidata_url]
    return wikidata_url.split('/')[-1]


def generate_stage1_data(triples, entity_indices, levels,
                          id2entity, id2relation,
                          one_hop=None,
                          token_prefix="[MM", token_suffix="]",
                          mode="tail",
                          use_readable_names=True,
                          url2name=None,
                          wikidata_relations=None,
                          use_entity_id=True,
                          use_candidates=True,
                          entities_need_unique_eid=None, entities_with_eid_not=None):
    
    data = []
    for h_id, r_id, t_id in tqdm(triples, desc="Generate Stage1 data", unit="strip"):
        h_code, _ = index_to_token_sequence(
            entity_indices[h_id], levels, token_prefix, token_suffix,
            entity_id=h_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        t_code, _ = index_to_token_sequence(
            entity_indices[t_id], levels, token_prefix, token_suffix,
            entity_id=t_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )

        h_url = id2entity.get(h_id, f"entity_{h_id}")
        t_url = id2entity.get(t_id, f"entity_{t_id}")
        rel_url = id2relation.get(r_id, f"relation_{r_id}")

        if use_readable_names:
            h_name = get_entity_name(h_url, url2name)
            t_name = get_entity_name(t_url, url2name)
            rel_name = get_relation_name(rel_url, wikidata_relations)
        else:
            h_name, t_name, rel_name = h_url, t_url, rel_url

        if mode in ["tail", "both"]:
            prompt = f"The query triplet is ({h_name}, {rel_name}, ?). \n"
            prompt += f"The quantized representation of entity {h_name} is: {h_code}\n"

            
            if use_candidates and one_hop is not None:
                candidates = one_hop.get(h_id, [])
                prompt += "The answer candidates and corresponding quantized representations are as follows: \n"
                
                neg_candidates = [c for c in candidates if c != t_id]
                import random
                sample_candidates = random.sample(neg_candidates, min(4, len(neg_candidates)))
                for c_id in sample_candidates:
                    c_code, _ = index_to_token_sequence(
                        entity_indices[c_id], levels, token_prefix, token_suffix,
                        entity_id=c_id if use_entity_id else None,
                        entities_need_unique_eid=entities_need_unique_eid,
                        entities_with_eid_not=entities_with_eid_not
                    )
                    c_name = get_entity_name(id2entity.get(c_id, f"entity_{c_id}"), url2name)
                    prompt += f"{c_name}: {c_code}\n"

            prompt += "Please generate the quantized representation of the answer entity:"
            prompt += f"\n{t_code}"
            data.append(prompt)

    logger.info(f"Stage 1 数据生成完成: {len(data)} 条 (mode={mode})")
    return data


def generate_stage2_data(triples, entity_indices, levels,
                          id2entity, id2relation,
                          one_hop=None,
                          token_prefix="[MM", token_suffix="]",
                          mode="both",
                          use_readable_names=True,
                          url2name=None,
                          wikidata_relations=None,
                          use_entity_id=True,
                          use_candidates=True,
                          entities_need_unique_eid=None, entities_with_eid_not=None):
    
    data = []
    for h_id, r_id, t_id in tqdm(triples, desc="Generate Stage2 data", unit="strip"):
        h_code, _ = index_to_token_sequence(
            entity_indices[h_id], levels, token_prefix, token_suffix,
            entity_id=h_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        t_code, _ = index_to_token_sequence(
            entity_indices[t_id], levels, token_prefix, token_suffix,
            entity_id=t_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )

        h_url = id2entity.get(h_id, f"entity_{h_id}")
        t_url = id2entity.get(t_id, f"entity_{t_id}")
        rel_url = id2relation.get(r_id, f"relation_{r_id}")

        if use_readable_names:
            h_name = get_entity_name(h_url, url2name)
            t_name = get_entity_name(t_url, url2name)
            rel_name = get_relation_name(rel_url, wikidata_relations)
        else:
            h_name, t_name, rel_name = h_url, t_url, rel_url

        if mode in ["tail", "both"]:
            prompt = f"The query triplet is ({h_name}, {rel_name}, ?). \n"
            prompt += f"The quantized representation of entity {h_name} is: {h_code}\n"

            
            if use_candidates and one_hop is not None:
                candidates = one_hop.get(h_id, [])
                prompt += "The answer candidates and corresponding quantized representations are as follows: \n"
                
                neg_candidates = [c for c in candidates if c != t_id]
                import random
                sample_candidates = random.sample(neg_candidates, min(4, len(neg_candidates)))
                for c_id in sample_candidates:
                    c_code, _ = index_to_token_sequence(
                        entity_indices[c_id], levels, token_prefix, token_suffix,
                        entity_id=c_id if use_entity_id else None,
                        entities_need_unique_eid=entities_need_unique_eid,
                        entities_with_eid_not=entities_with_eid_not
                    )
                    c_name = get_entity_name(id2entity.get(c_id, f"entity_{c_id}"), url2name)
                    prompt += f"{c_name}: {c_code}\n"

            prompt += "Please generate the quantized representation of the answer entity:"
            prompt += f"\n{t_code}"
            data.append(prompt)

    logger.info(f"Stage 2 数据生成完成: {len(data)} 条 (mode={mode})")
    return data


def generate_test_prompts(triples, entity_indices, levels,
                           id2entity, id2relation,
                           one_hop=None,
                           token_prefix="[MM", token_suffix="]",
                           mode="both",
                           use_readable_names=True,
                           url2name=None,
                           wikidata_relations=None,
                           use_entity_id=True,
                           entities_need_unique_eid=None, entities_with_eid_not=None):
    
    prompts = []
    answers = []
    meta = []

    for h_id, r_id, t_id in tqdm(triples, desc="Generate test prompts", unit="strip"):
        h_code, _ = index_to_token_sequence(
            entity_indices[h_id], levels, token_prefix, token_suffix,
            entity_id=h_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        t_code, _ = index_to_token_sequence(
            entity_indices[t_id], levels, token_prefix, token_suffix,
            entity_id=t_id if use_entity_id else None,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        
        h_url = id2entity.get(h_id, f"entity_{h_id}")
        t_url = id2entity.get(t_id, f"entity_{t_id}")
        rel_url = id2relation.get(r_id, f"relation_{r_id}")
        
        if use_readable_names:
            h_name = get_entity_name(h_url, url2name)
            t_name = get_entity_name(t_url, url2name)
            rel_name = get_relation_name(rel_url, wikidata_relations)
        else:
            h_name, t_name, rel_name = h_url, t_url, rel_url

        if mode in ["tail", "both"]:
            
            
            prompt = f"The query triplet is ({h_name}, {rel_name}, ?). \n"
            prompt += f"The quantized representation of entity {h_name} is: {h_code}\n"
            prompt += "Please generate the quantized representation of the answer entity:"

            prompts.append(prompt)
            answers.append(t_code)
            meta.append({
                "h_id": h_id, "r_id": r_id, "t_id": t_id,
                "h_name": h_name, "rel_name": rel_name,
                "mode": "tail", "answer_entity_id": t_id,
            })

    return prompts, answers, meta


def main():
    parser = argparse.ArgumentParser(description="生成 LLM 训练数据集")
    parser.add_argument("--dataset", type=str, default="MKG-W", choices=["MKG-W", "MKG-Y", "MMKB-DB15K"])
    parser.add_argument("--data_path", type=str, default=None, help="数据集路径，默认从default_config读取")
    parser.add_argument("--fsq_dir", type=str, default="./outputs/fsq_model/MKG-W")
    parser.add_argument("--output_dir", type=str, default="./outputs/llm_datasets")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 8, 8, 8, 8, 8])
    parser.add_argument("--token_prefix", type=str, default="[MM")
    parser.add_argument("--token_suffix", type=str, default="]")
    parser.add_argument("--use_neighborhood", action="store_true", default=True)
    parser.add_argument("--use_entity_id", action="store_true", default=True, help="添加实体ID token以保证唯一性")
    parser.add_argument("--only_test", action="store_true", default=False, help="只生成测试集")

    args = parser.parse_args()

    
    fsq_config_path = os.path.join(args.fsq_dir, 'config.json')
    if os.path.exists(fsq_config_path):
        with open(fsq_config_path) as f:
            fsq_config = json.load(f)
        args.levels = fsq_config.get('levels', args.levels)
        logger.info(f"从 FSQ 配置加载 levels: {args.levels}")

    
    entity2id_path = os.path.join(args.fsq_dir, 'entity2id.json')
    relation2id_path = os.path.join(args.fsq_dir, 'relation2id.json')
    id2entity_path = os.path.join(args.fsq_dir, 'id2entity.json')
    id2relation_path = os.path.join(args.fsq_dir, 'id2relation.json')

    if os.path.exists(entity2id_path):
        with open(entity2id_path) as f:
            entity2id = json.load(f)
        with open(relation2id_path) as f:
            relation2id = json.load(f)
        with open(id2entity_path) as f:
            id2entity = {int(k): v for k, v in json.load(f).items()}
        with open(id2relation_path) as f:
            id2relation = {int(k): v for k, v in json.load(f).items()}
    else:
        from configs.default_config import DATASET_CONFIG
        data_path = DATASET_CONFIG[args.dataset]["data_path"]
        entity2id, relation2id, id2entity, id2relation = build_entity_relation_ids(data_path)

    
    indices_path = os.path.join(args.fsq_dir, 'entity_fsq_indices.npy')
    if os.path.exists(indices_path):
        fsq_indices = np.load(indices_path)
        entity_indices = {i: int(fsq_indices[i]) for i in range(len(fsq_indices))}
    else:
        logger.warning("未找到 FSQ 索引文件，使用随机索引")
        codebook_size = 1
        for l in args.levels:
            codebook_size *= l
        entity_indices = {i: random.randint(0, codebook_size - 1) for i in range(len(entity2id))}

    num_entities = len(entity2id)

    
    duplicate_codes, entities_need_unique_eid, entities_with_eid_not = find_duplicate_entities(entity_indices)
    logger.info(f"FSQ码字分析: {len(duplicate_codes)} 个码字存在碰撞")
    logger.info(f"  - {len(entities_need_unique_eid)} 个冲突实体需要唯一EID")
    logger.info(f"  - {len(entities_with_eid_not)} 个非冲突实体使用 [EID_NOT]")

    
    if args.use_entity_id:
        
        all_tokens, token2id = get_all_tokens(
            args.levels, num_entities,
            args.token_prefix, args.token_suffix,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        logger.info(f"新 token 数量: {len(all_tokens)} (FSQ tokens + 1 [EID_NOT] + {len(entities_need_unique_eid)} 唯一EID)")
    else:
        all_tokens, token2id = get_all_tokens(args.levels, num_entities, args.token_prefix, args.token_suffix)
        logger.info(f"新 token 数量: {len(all_tokens)} (仅FSQ tokens)")

    
    from configs.default_config import DATASET_CONFIG
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = DATASET_CONFIG[args.dataset]["data_path"]

    
    if args.only_test:
        test_triples = read_triples(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
        logger.info(f"测试三元组数量: {len(test_triples)}")

        
        one_hop = None
        if args.use_neighborhood:
            neighborhood_path = os.path.join(data_path, "one_hop.json")
            if os.path.exists(neighborhood_path):
                one_hop = load_neighborhood(neighborhood_path)

        
        url2name = load_entity_names(data_path)
        
        wikidata_relations = load_relation_names(data_path)

        os.makedirs(args.output_dir, exist_ok=True)

        test_prompts, test_answers, test_meta = generate_test_prompts(
            test_triples, entity_indices, args.levels,
            id2entity, id2relation, one_hop,
            args.token_prefix, args.token_suffix, mode="both",
            use_readable_names=True, url2name=url2name,
            wikidata_relations=wikidata_relations,
            use_entity_id=args.use_entity_id,
            entities_need_unique_eid=entities_need_unique_eid,
            entities_with_eid_not=entities_with_eid_not
        )
        with open(os.path.join(args.output_dir, 'test_prompts.json'), 'w') as f:
            json.dump({'prompts': test_prompts, 'answers': test_answers, 'meta': test_meta}, f, ensure_ascii=False)
        logger.info(f"测试数据已保存 (仅测试集)")
        
        num_unique_eid = len(entities_need_unique_eid) if args.use_entity_id else 0
        token_info = {
            'tokens': all_tokens,
            'token2id': token2id,
            'levels': args.levels,
            'num_unique_eid': num_unique_eid,
            'entities_need_unique_eid': list(entities_need_unique_eid),
            'entities_with_eid_not': list(entities_with_eid_not)
        }
        with open(os.path.join(args.output_dir, 'new_tokens.json'), 'w') as f:
            json.dump(token_info, f, ensure_ascii=False)
        logger.info("Token 信息已保存")
        return

    train_triples = read_triples(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triples(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triples(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    logger.info(f"三元组数量: train={len(train_triples)}, valid={len(valid_triples)}, test={len(test_triples)}")

    
    one_hop = None
    if args.use_neighborhood:
        neighborhood_path = os.path.join(data_path, "one_hop.json")
        if os.path.exists(neighborhood_path):
            one_hop = load_neighborhood(neighborhood_path)
        else:
            logger.warning(f"未找到一阶邻居文件: {neighborhood_path}")

    
    url2name = load_entity_names(data_path)

    
    wikidata_relations = load_relation_names(data_path)

    
    os.makedirs(args.output_dir, exist_ok=True)

    
    stage1_triples = train_triples + valid_triples
    logger.info(f"Stage 1 使用 {len(stage1_triples)} 三元组 (train + valid)")
    stage1_data = generate_stage1_data(
        stage1_triples, entity_indices, args.levels,
        id2entity, id2relation, one_hop,
        args.token_prefix, args.token_suffix, mode="tail",
        use_readable_names=True, url2name=url2name,
        wikidata_relations=wikidata_relations,
        use_entity_id=args.use_entity_id,
        use_candidates=True,  
        entities_need_unique_eid=entities_need_unique_eid,
        entities_with_eid_not=entities_with_eid_not
    )
    with open(os.path.join(args.output_dir, 'stage1_token_data.json'), 'w') as f:
        json.dump(stage1_data, f, ensure_ascii=False)
    logger.info(f"Stage 1 数据已保存")

    
    stage2_train = generate_stage2_data(
        train_triples, entity_indices, args.levels,
        id2entity, id2relation, one_hop,
        args.token_prefix, args.token_suffix, mode="tail",
        use_readable_names=True, url2name=url2name,
        wikidata_relations=wikidata_relations,
        use_entity_id=args.use_entity_id,
        use_candidates=True,  
        entities_need_unique_eid=entities_need_unique_eid,
        entities_with_eid_not=entities_with_eid_not
    )
    with open(os.path.join(args.output_dir, 'stage2_train_data.json'), 'w') as f:
        json.dump(stage2_train, f, ensure_ascii=False)
    logger.info(f"Stage 2 训练数据已保存")

    
    stage2_valid = generate_stage2_data(
        valid_triples, entity_indices, args.levels,
        id2entity, id2relation, one_hop,
        args.token_prefix, args.token_suffix, mode="both",
        use_readable_names=True, url2name=url2name,
        wikidata_relations=wikidata_relations,
        use_entity_id=args.use_entity_id,
        use_candidates=False,  
        entities_need_unique_eid=entities_need_unique_eid,
        entities_with_eid_not=entities_with_eid_not
    )
    with open(os.path.join(args.output_dir, 'stage2_valid_data.json'), 'w') as f:
        json.dump(stage2_valid, f, ensure_ascii=False)

    
    test_prompts, test_answers, test_meta = generate_test_prompts(
        test_triples, entity_indices, args.levels,
        id2entity, id2relation, one_hop,
        args.token_prefix, args.token_suffix, mode="both",
        use_readable_names=True, url2name=url2name,
        wikidata_relations=wikidata_relations,
        use_entity_id=args.use_entity_id,
        entities_need_unique_eid=entities_need_unique_eid,
        entities_with_eid_not=entities_with_eid_not
    )
    with open(os.path.join(args.output_dir, 'test_prompts.json'), 'w') as f:
        json.dump({'prompts': test_prompts, 'answers': test_answers, 'meta': test_meta}, f, ensure_ascii=False)
    logger.info(f"测试数据已保存")

    
    num_unique_eid = len(entities_need_unique_eid) if args.use_entity_id else 0
    token_info = {
        'tokens': all_tokens,
        'token2id': token2id,
        'levels': args.levels,
        'use_entity_id': args.use_entity_id,
        'num_entities': num_entities,
        'num_fsq_tokens': len(args.levels) * args.levels[0],
        'num_entity_id_tokens': num_unique_eid + 1,  
        'num_duplicate_codes': len(duplicate_codes),
        'entities_need_unique_eid': list(entities_need_unique_eid),  
        'entities_with_eid_not': list(entities_with_eid_not),  
        'eid_not_token': '[EID_NOT]'
    }
    with open(os.path.join(args.output_dir, 'new_tokens.json'), 'w') as f:
        json.dump(token_info, f, ensure_ascii=False)
    logger.info(f"Token 信息已保存: use_entity_id={args.use_entity_id}, num_entities={num_entities}")


if __name__ == "__main__":
    main()
