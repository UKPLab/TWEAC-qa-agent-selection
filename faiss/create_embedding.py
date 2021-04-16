import logging
import os
import argparse
import json
from elasticsearch import Elasticsearch
from numpy import linalg as LA
import numpy as np
from sentence_transformers import SentenceTransformer

from use_encoder import USEEncoderAPI, USEEncoder
logging.basicConfig()
logger = logging.getLogger("faiss")
logger.setLevel(logging.INFO)

def create_new(config, embedder):
    es = Elasticsearch([config["es_url"]], scheme="http")
    doc = {
        'size': min(10000, config["chunk"]),
        'query': {
            'match_all': {}
        },
        'sort': [
            {'question_idx': 'asc'}
        ]
    }
    os.makedirs(config["out_dir"], exist_ok=True)
    res = es.search(index=config["index_name"], body=doc, scroll='30m')
    idx = 0
    data = []
    while res["hits"]["hits"]:
        logger.info("Fetching next scroll from ES")
        data.extend([r["_source"]["question_text"] for r in res["hits"]["hits"]])
        if len(data) >= config["chunk"]:
            embeddings = embed(data[:config["chunk"]], embedder, config)
            embeddings = embeddings / LA.norm(embeddings, axis=0)
            np.save(os.path.join(config["out_dir"], "faiss_embedding_{}.npy".format(idx)), embeddings)
            idx += 1
            data = data[config["chunk"]:]
        scroll = res["_scroll_id"]
        res = es.scroll(scroll_id=scroll, scroll='30m')
    embeddings = embed(data, embedder, config)
    embeddings = embeddings / LA.norm(embeddings, axis=0)
    np.save(os.path.join(config["out_dir"], "faiss_embedding_{}.npy".format(idx)), embeddings)

def embed(sentences, embedder, config):
    logger.info("Embedding chunk")
    return np.array(embedder.encode(sentences, batch_size=config.get("batchsize", 32)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if "cache_dir" in config:
        os.environ['TORCH_HOME'] = config["cache_dir"]
    if config["sentence_transformer_model"] == "use-qa":
        embedder = USEEncoderAPI()
    elif "https" in config["sentence_transformer_model"]:
        embedder = USEEncoder(config["sentence_transformer_model"])
    else:
        embedder = SentenceTransformer(config["sentence_transformer_model"])

    create_new(config, embedder)