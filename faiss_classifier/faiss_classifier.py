import logging
import os
import argparse
import json
from collections import defaultdict
import datetime
from elasticsearch import Elasticsearch
from progressbar import progressbar
import numpy as np
from sentence_transformers import SentenceTransformer

import faiss
from use_encoder import USEEncoder, USEEncoderAPI

logging.basicConfig()
logger = logging.getLogger("faiss")
logger.setLevel(logging.INFO)

def eval_model(index, embedder, es, count, config, batch_query, out_result=True):
    logger.info(config)
    base_path = config.get("base_path", "")
    results_all = defaultdict(lambda: list())
    confusion_matrix = [[0]*len(count) for _ in count]
    activation_matrix = [[0]*len(count) for _ in count]
    agents = list(config["agents"])
    for i, (agent, path) in enumerate(config["agents"].items()):
        logger.info("Evaluating agent {}/{}: {}".format(i+1, len(config["agents"]), agent))
        logger.info("Reading questions from {}".format(agent, path))
        data = _read_file(path, base_path, config.get("truncate", -1))
        logger.info("{} questions".format(len(data)))
        start = datetime.datetime.now()
        ranks = []
        average_precisions = []
        activation = np.zeros(len(count))
        vectors = data
        if batch_query:
            vectors = embedder.encode(data)
            fake_embedder = type('embedder', (), {})()
            fake_embedder.encode = lambda sents, show_progress_bar: sents
        else:
            fake_embedder = embedder
        for query in progressbar(vectors):
            res = inference(index, fake_embedder, es, query, count, config)
            rank = 0
            precisions = 0
            for j, (answer, score) in enumerate(res, start=1):
                idx = agents.index(answer)
                if j==1:
                    confusion_matrix[i][idx] += 1
                activation[idx] += score
                if answer == agent:
                    if rank == 0:
                        rank = j
                    precisions = 1 / float(j)

            ranks.append(rank)
            average_precisions.append(precisions)
        end = datetime.datetime.now()
        time_taken = end - start
        correct_answers = len([a for a in ranks if a == 1])
        recall_3 = len([a for a in ranks if a <= 3 and a != 0])
        recall_5 = len([a for a in ranks if a <= 5 and a != 0])
        results = {
            'accuracy': correct_answers / float(len(ranks)),
            'mrr': np.mean([1 / float(r) if r>0 else 0 for r in ranks]),
            'r@3': recall_3/float(len(ranks)),
            'r@5': recall_5/float(len(ranks)),
            'query_per_second': float(len(ranks))/time_taken.total_seconds()
        }
        activation = (activation/float(len(ranks))).tolist()
        activation_matrix[i] = activation
        for key, value in results.items():
            results_all[key].append(value)
        results_all[agent] = results

    results_all["confusion_matrix"] = confusion_matrix
    results_all["activation_matrix"] = activation_matrix
    precision, recall, f1 = _compute_f1(confusion_matrix)
    for idx, agent in enumerate(config["agents"].keys()):
        results = results_all[agent]
        results["precision"] = precision[idx]
        results["recall"] = recall[idx]
        results["f1"] = f1[idx]
        results_all["precision"].append(precision[idx])
        results_all["recall"].append(recall[idx])
        results_all["f1"].append(f1[idx])

        logger.info('\nResults for agent {}'.format(agent))
        _log_results(results)

    for key in ["accuracy", "precision", "recall", "f1", "mrr", "r@3", 'r@5', "query_per_second"]:
        results_all[key] = np.mean(results_all[key])
    logger.info('\nResults for all datasets:')
    _log_results(results_all)

    if config["out_dir"] and out_result:
        folder_name = config.get("folder_prefix", "")+"_"
        path = os.path.join(config["out_dir"], folder_name+datetime.datetime.now().strftime("%b-%d-%Y_%H-%M"))
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(path, "results.json"), "w") as f:
            json.dump(results_all, f, indent=4)
    return results_all


def _log_results(results):
    logger.info('Accuracy: {}'.format(results['accuracy']))
    logger.info('Precision: {}'.format(results['precision']))
    logger.info('Recall: {}'.format(results['recall']))
    logger.info('F1: {}'.format(results['f1']))
    logger.info('MRR: {}'.format(results['mrr']))
    logger.info('R@3: {}'.format(results['r@3']))
    logger.info('R@5: {}'.format(results['r@5']))
    logger.info('Queries/second: {}'.format(results['query_per_second']))


def _compute_f1(confusion_matrix):
    matrix = np.array(confusion_matrix)
    relevant = np.sum(matrix, axis=1)
    retrieved = np.sum(matrix, axis=0)
    precision, recall, f1 = [], [], []
    for i, val in enumerate(np.diag(matrix)):
        if retrieved[i]==0:
            p=0
        else:
            p = val/retrieved[i]
        if relevant[i]==0:
            r=0
        else:
            r = val/relevant[i]
        precision.append(p)
        recall.append(r)
        if r==0 or p==0:
            f1.append(0)
        else:
            f1.append((2*r*p)/(r+p))
    return precision, recall, f1

def inference(index, embedder, es, query, count, config):
    embedding = embedder.encode(query)[0]
    similar_embeddings, similar_items = index.search(embedding.reshape(1, -1), config["k"])
    similar_items = similar_items[0]

    body = {
        "query": {
            "bool": {
                "filter": {
                    "terms": {"question_idx": similar_items}
                }
            }
        }
    }
    res = es.search(index=config["index_name"], body=body)["hits"]["hits"]
    if config["weighting"] == "uniform":
        weights = [1/len(res)]*len(res)
    elif config["weighting"] == "score":
        weights = [r["_score"] for r in res]
    votes = {}
    for r, w in zip(res, weights):
        rs = r["_source"]
        if config.get("class_weighting", False):
            w = w/count[rs["agent"]]
        if rs["agent"] in votes:
            votes[rs["agent"]] += w
        else:
            votes[rs["agent"]] = w
    total = sum(votes.values())
    votes = {key: votes[key]/total for key in votes}
    score = sorted(list(votes.items()), key=lambda v: v[1], reverse=True)
    return score


def _read_file(paths, base_paths, truncate=-1):
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
        data.append(sentences[i])
    return data

def build_index(config):
    index = faiss.IndexFlatIP(config["embedding_dim"])
    num_chunks = len([file for file in os.listdir(config["embedding_dir"]) if ".npy" in file])
    for i in range(num_chunks):
        embs = np.load(os.path.join(config["embedding_dir"], "faiss_embedding_{}.npy".format(i)))
        index.add(embs)
    if config["use_gpu"]:
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def parameter_grid_search(index, es, embedder, config, count, batch_query):
    ks = config["k"]
    weightings = config["weighting"]
    eval_config = config.copy()
    results = []
    for k in ks:
        for weighting in weightings:
            eval_config["k"] = k
            eval_config["weighting"] = weighting
            results.append({"k": k, "weighting": weighting, "result": eval_model(index, embedder, es, count, eval_config, batch_query,out_result=False)})
    sorted_res = sorted(results, reverse=True, key=lambda res: res["result"]["accuracy"])
    for res in sorted_res:
        logger.info("k: {}\tweighting: {}\tuse cosine: {}\taccuracy: {}".format(res["k"], res["weighting"], res["use_cosine"], res["result"]["accuracy"]))
    if config["out_dir"]:
        folder_name = config.get("folder_prefix", "")+"_"
        path = os.path.join(config["out_dir"], folder_name+datetime.datetime.now().strftime("%b-%d-%Y_%H-%M"))
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        with open(os.path.join(path, "grid_search_results.json"), "w") as f:
            json.dump(sorted_res, f, indent=4)
    return sorted_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.json")
    parser.add_argument("--query", type=str)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--grid_search", action='store_true')
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--batch_query", default=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    es = Elasticsearch([config["es_url"]], scheme="http")

    res = es.search(index=config["index_name"], body={
        "aggs" : {
            "agents" : {
                "terms" : {
                    "field": "agent",
                    "size": 300
                }
            }
        }
    })["aggregations"]["agents"]
    count = {b["key"]: b["doc_count"] for b in res["buckets"]}
    index = build_index(config)
    if "cache_dir" in config:
        os.environ['TORCH_HOME'] = config["cache_dir"]
    if "cache_dir" in config:
        os.environ['TORCH_HOME'] = config["cache_dir"]
    if config["sentence_transformer_model"] == "use-qa":
        embedder = USEEncoderAPI()
    elif "https" in config["sentence_transformer_model"]:
        embedder = USEEncoder(config["sentence_transformer_model"])
    else:
        embedder = SentenceTransformer(config["sentence_transformer_model"])
    if args.inference:
        inference(index, embedder, es, args.query, count, config)
    elif args.eval:
        eval_model(index, embedder, es, count, config, args.batch_query)
    elif args.grid_search:
        parameter_grid_search(index, es, embedder, config, count, args.batch_query)