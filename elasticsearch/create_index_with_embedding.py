import logging
import os
import argparse
import json
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import numpy as np
import base64

from es_vec_classifier.use_encoder import USEEncoderAPI, USEEncoder

np.random.seed(123456789) # makes random sampling from training data deterministic between runs

logging.basicConfig()
logger = logging.getLogger("es")
logger.setLevel(logging.INFO)

def create_new(config, embedder):
    _create_index_request_body = {
        "mappings": {
            "properties": {
                "question_idx": {
                    "type": "integer"
                },
                "question_text": {
                    "type": "text",
                },
                "agent": {
                    "type": "keyword",
                },
                "embedding_vector": {
                    "type": "binary",
                    "doc_values": True
                }
            }
        }
    }
    es = Elasticsearch([config["es_url"]], scheme="http")
    es.cluster.health(wait_for_status='yellow', request_timeout=60*20)
    es.indices.delete(index=config["index_name"], ignore=[400, 404], request_timeout=5*60)
    es.indices.create(index=config["index_name"], body=_create_index_request_body)

    base_path = config.get("base_path", "")
    idx = 0
    all_data = []
    truncate_idx = 0
    for (agent, path) in config["agents"].items():
        logger.info("Reading questions for agent {} from {}".format(agent, path))
        truncate = config.get("truncate", -1)
        if isinstance(truncate, list):
            truncate = truncate[truncate_idx]
            truncate_idx = (truncate_idx+1)%len(config.get("truncate", -1))
        data, idx = _read_file(path, base_path, idx, agent, config["index_name"], truncate)
        data = _embed_data(data, embedder, config)
        logger.info("{} questions".format(len(data)))
        all_data.extend(data)
    success, failed = bulk(es, all_data, request_timeout=5*60, max_retries=5)
    logger.info('Creating index "{}"... (Success={}, Failed={})'.format(config["index_name"], success, failed))


def add(config, embedder):
    es = Elasticsearch([config["es_url"]], scheme="http")

    base_path = config.get("base_path", "")
    res = es.count(index=config["index_name"])
    idx = res["count"]
    all_data = []
    for (agent, path) in config["agents"].items():
        logger.info("Reading questions for agent {} from {}".format(agent, path))
        data, idx = _read_file(path, base_path, idx, agent, config["index_name"], config.get("truncate", -1))
        data = _embed_data(data, embedder, config)
        logger.info("{} questions".format(len(data)))
        all_data.extend(data)
    success, failed = bulk(es, all_data, request_timeout=5*60, max_retries=5)
    logger.info('Adding to index "{}"... (Success={}, Failed={})'.format(config["index_name"], success, failed))


def _embed_data(data, embedder, config):
    arr = _embed_and_encode([d["_source"]["question_text"] for d in data], embedder, config)
    for d, vec in zip(data, arr):
        d["_source"]["embedding_vector"] = vec
    return data


def _embed_and_encode(sentences, embedder, config):
    def encode(vec):
        dfloat32 = np.dtype(">f4") # big endian 32bit float
        base64_str = base64.b64encode(np.array(vec).astype(dfloat32)).decode("utf-8")
        return base64_str
    logger.info("Embedding chunk")
    arr = embedder.encode(sentences, batch_size=config.get("batchsize", 32))
    arr = [encode(vec) for vec in arr]
    return arr

def _read_file(paths, base_paths, idx, agent, index, truncate=-1):
    if not isinstance(paths, list):
        paths = [paths]
    if not isinstance(base_paths, list):
        base_paths = [base_paths]
    data = []
    sentences = []
    for path in paths:
        for base_path in base_paths:
            entire_path = os.path.join(base_path, path)
            if os.path.isfile(entire_path):
                exists = True
                with open(entire_path, "r", encoding="utf-8") as f:
                    for l in f.readlines():
                        sentences.append(l.strip())
                    if not exists:
                        raise FileNotFoundError("{} was not found in any base path".format(path))
    chosen_idxs = np.random.choice(len(sentences), truncate) if truncate > 0 else range(len(sentences))
    for i in chosen_idxs:
        data.append({
            "_index": index,
            "_source": {
                'question_idx': idx,
                'question_text': sentences[i],
                'agent': agent
            }
        })
        idx += 1

    return data, idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml")
    parser.add_argument("--add", action='store_true')
    parser.add_argument("--new", action='store_true')
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.load(f)
    if "cache_dir" in config:
        os.environ['TORCH_HOME'] = config["cache_dir"]
    if config["sentence_transformer_model"] == "use-qa":
        embedder = USEEncoderAPI()
    elif "https" in config["sentence_transformer_model"]:
        embedder = USEEncoder(config["sentence_transformer_model"])
    else:
        embedder = SentenceTransformer(config["sentence_transformer_model"])
    if args.new:
        create_new(config, embedder)
    elif args.add:
        add(config, embedder)
    else:
        logger.info("Add either --new or --add")