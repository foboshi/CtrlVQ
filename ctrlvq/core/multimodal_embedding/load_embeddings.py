

import os
import logging
import numpy as np
import torch
import h5py
from collections import OrderedDict

logger = logging.getLogger(__name__)


def read_triples(file_path, entity2id, relation2id):
    
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h in entity2id and r in relation2id and t in entity2id:
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def build_entity_relation_ids(data_path):
    
    ent_set = OrderedDict()
    rel_set = OrderedDict()

    for split in ['train', 'valid', 'test']:
        filepath = os.path.join(data_path, f'{split}.txt')
        if not os.path.exists(filepath):
            logger.warning(f"文件不存在: {filepath}")
            continue
        with open(filepath, encoding='utf-8') as fin:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                sub, rel, obj = parts
                if sub not in ent_set:
                    ent_set[sub] = len(ent_set)
                if rel not in rel_set:
                    rel_set[rel] = len(rel_set)
                if obj not in ent_set:
                    ent_set[obj] = len(ent_set)

    entity2id = dict(ent_set)
    relation2id = dict(rel_set)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    logger.info(f"构建完成: {len(entity2id)} 个实体, {len(relation2id)} 个关系")
    return entity2id, relation2id, id2entity, id2relation


def load_entity_links_mkg(ent_link_path):
    
    ent_link = {}
    with open(ent_link_path, encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                dbpedia_uri, target_uri = parts
                ent_link[dbpedia_uri] = target_uri
    logger.info(f"加载了 {len(ent_link)} 条实体链接 (MKG)")
    return ent_link


def load_entity_links_mmkb(ent_link_path):
    
    link_fb_db = {}
    with open(ent_link_path, encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split(' ')
            if len(parts) >= 3:
                freebase_id = parts[0]
                dbpedia_uri = parts[2]
                
                if dbpedia_uri.startswith('<') and dbpedia_uri.endswith('>'):
                    dbpedia_uri = dbpedia_uri[1:-1]
                link_fb_db[freebase_id] = dbpedia_uri
    logger.info(f"加载了 {len(link_fb_db)} 条实体链接 (MMKB)")
    return link_fb_db


def load_multimodal_embeddings_mkg(text_path, image_path, ent_link_path,
                                    entity2id, device='cpu',
                                    text_max_sentences=4, text_emb_dim=384,
                                    img_max_patches=24, img_emb_dim=383):
    
    num_entities = len(entity2id)
    ent_text_emb = torch.zeros([num_entities, text_max_sentences, text_emb_dim],
                                device=device)
    ent_img_emb = torch.zeros([num_entities, img_max_patches, img_emb_dim],
                               device=device)

    
    ent_link = load_entity_links_mkg(ent_link_path)

    
    text_count = 0
    text_loaded = 0
    if os.path.exists(text_path):
        with h5py.File(text_path, 'r') as f:
            for k in f.keys():
                v = np.array(f[k])
                sentence_num = v.shape[0]
                try:
                    entity_name = ent_link.get('http://dbpedia.org/resource/' + k)
                    if entity_name and entity_name in entity2id:
                        eid = entity2id[entity_name]
                        if sentence_num >= text_max_sentences:
                            ent_text_emb[eid] = torch.from_numpy(
                                v[:text_max_sentences])
                        else:
                            ent_text_emb[eid][:sentence_num] = torch.from_numpy(v)
                        text_loaded += 1
                except (KeyError, ValueError):
                    text_count += 1
    else:
        logger.warning(f"文本嵌入文件不存在: {text_path}")

    
    image_count = 0
    image_loaded = 0
    if os.path.exists(image_path):
        with h5py.File(image_path, 'r') as f:
            for k in f.keys():
                v = np.array(f[k])
                try:
                    entity_name = ent_link.get('http://dbpedia.org/resource/' + k)
                    if entity_name and entity_name in entity2id:
                        eid = entity2id[entity_name]
                        ent_img_emb[eid] = torch.from_numpy(v)
                        image_loaded += 1
                except (KeyError, ValueError):
                    image_count += 1
    else:
        logger.warning(f"图像嵌入文件不存在: {image_path}")

    logger.info(f"MKG 多模态嵌入加载完成: "
                f"文本={text_loaded}/{num_entities} (缺失映射={text_count}), "
                f"图像={image_loaded}/{num_entities} (缺失映射={image_count})")

    return ent_text_emb, ent_img_emb


def load_multimodal_embeddings_mmkb(text_path, image_path, ent_link_path,
                                     entity2id, device='cpu',
                                     text_max_sentences=4, text_emb_dim=384,
                                     img_max_patches=24, img_emb_dim=383):
    
    num_entities = len(entity2id)
    ent_text_emb = torch.zeros([num_entities, text_max_sentences, text_emb_dim],
                                device=device)
    ent_img_emb = torch.zeros([num_entities, img_max_patches, img_emb_dim],
                               device=device)

    
    link_fb_db = load_entity_links_mmkb(ent_link_path)

    
    text_count = 0
    text_loaded = 0
    if os.path.exists(text_path):
        with h5py.File(text_path, 'r') as f:
            for k in f.keys():
                v = np.array(f[k])
                sentence_num = v.shape[0]
                try:
                    name = 'http://dbpedia.org/resource/' + k
                    if name in entity2id:
                        eid = entity2id[name]
                        if sentence_num >= text_max_sentences:
                            ent_text_emb[eid] = torch.from_numpy(
                                v[:text_max_sentences])
                        else:
                            ent_text_emb[eid][:sentence_num] = torch.from_numpy(v)
                        text_loaded += 1
                except (KeyError, ValueError):
                    text_count += 1
    else:
        logger.warning(f"文本嵌入文件不存在: {text_path}")

    
    image_count = 0
    image_loaded = 0
    if os.path.exists(image_path):
        with h5py.File(image_path, 'r') as f:
            for k in f.keys():
                v = np.array(f[k])
                try:
                    name = link_fb_db.get('/m/' + k[2:])
                    if name and name in entity2id:
                        eid = entity2id[name]
                        ent_img_emb[eid] = torch.from_numpy(v)
                        image_loaded += 1
                except (KeyError, ValueError):
                    image_count += 1
    else:
        logger.warning(f"图像嵌入文件不存在: {image_path}")

    logger.info(f"MMKB 多模态嵌入加载完成: "
                f"文本={text_loaded}/{num_entities} (缺失映射={text_count}), "
                f"图像={image_loaded}/{num_entities} (缺失映射={image_count})")

    return ent_text_emb, ent_img_emb


def load_structural_embeddings(structural_emb_path, entity2id, device='cpu'):
    
    num_entities = len(entity2id)

    if not os.path.exists(structural_emb_path):
        logger.warning(f"结构嵌入文件不存在: {structural_emb_path}")
        
        return torch.zeros([num_entities, 0], device=device)

    
    struct_emb = np.load(structural_emb_path)
    logger.info(f"加载结构嵌入: 形状 {struct_emb.shape}")

    
    
    ent_struct_emb = torch.from_numpy(struct_emb).float()

    
    if ent_struct_emb.shape[0] != num_entities:
        logger.warning(
            f"结构嵌入实体数 ({ent_struct_emb.shape[0]}) 与 "
            f"entity2id 数量 ({num_entities}) 不匹配!"
        )
        
        if ent_struct_emb.shape[0] > num_entities:
            ent_struct_emb = ent_struct_emb[:num_entities]
        else:
            padding = torch.zeros(
                [num_entities - ent_struct_emb.shape[0], ent_struct_emb.shape[1]]
            )
            ent_struct_emb = torch.cat([ent_struct_emb, padding], dim=0)

    return ent_struct_emb.to(device)


def load_multimodal_embeddings(dataset_name, dataset_config, entity2id,
                                device='cpu', structural_emb_path=None):
    
    cfg = dataset_config
    common_kwargs = dict(
        text_path=cfg["text_emb_path"],
        image_path=cfg["image_emb_path"],
        ent_link_path=cfg["ent_link_path"],
        entity2id=entity2id,
        device=device,
        text_max_sentences=cfg["text_max_sentences"],
        text_emb_dim=cfg["text_emb_dim"],
        img_max_patches=cfg["img_max_patches"],
        img_emb_dim=cfg["img_emb_dim"],
    )

    if cfg["link_type"] == "mkg":
        ent_text_emb, ent_img_emb = load_multimodal_embeddings_mkg(**common_kwargs)
    elif cfg["link_type"] == "mmkb":
        ent_text_emb, ent_img_emb = load_multimodal_embeddings_mmkb(**common_kwargs)
    else:
        raise ValueError(f"不支持的链接类型: {cfg['link_type']}")

    
    ent_struct_emb = None
    if structural_emb_path:
        ent_struct_emb = load_structural_embeddings(
            structural_emb_path, entity2id, device
        )

    return ent_text_emb, ent_img_emb, ent_struct_emb
