import torch
torch.cuda.current_device()
import argparse
import datetime
import json
from collections import defaultdict
import yaml
import logging
import os
import hashlib
import base64
from copy import deepcopy
import numpy as np
np.random.seed(123456789) # makes random sampling from training data deterministic between runs
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    ElectraConfig, ElectraTokenizer, ElectraModel,
    AlbertConfig, AlbertTokenizer, AlbertModel,
    get_linear_schedule_with_warmup,
)

from transformer_model import TransformerModel, TransformerModelV2, TransformerModelV3, TransformerModelSoftmax, \
    TransformerModelMSE, TransformerModelPairwise

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer)
}


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Manager:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.get("no_gpu", False) else "cpu")
        self.all_agents = config["all_agents"]
        self.agent_weights = [1.0] * len(self.all_agents)
        self.agent_weights_ce = [1.0] * len(self.all_agents)
        hashed_config = "" if config.get("disable_hash_in_folder_name", False) else \
            str(base64.b64encode(hashlib.md5(bytearray(json.dumps(config, sort_keys=True), encoding="utf-8"))
                                 .digest()), "utf-8").replace("\\'", "_").replace("/", "_")
        folder_name = "{}_{}_{}".format(config["model"]["model"], config["model"]["model_name"], hashed_config)

        if "folder_name_prefix" in config:
            folder_name = "{}_{}".format(config["folder_name_prefix"], folder_name)
        self.out_dir = os.path.join(config["out_dir"], folder_name)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.yaml"), mode="w") as f:
            yaml.dump(config, f)
        os.makedirs(os.path.join(self.out_dir, "latest_model"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "best_model"), exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.all_results = None
        if os.path.exists(os.path.join(self.out_dir, "results.json")):
            with open(os.path.join(self.out_dir, "results.json")) as f:
                self.all_results = json.load(f)
        else:
            self.all_results = {"best_epoch": -1, "latest_epoch": -1, "epoch_results": [], "test_results": {}}
        logger.info("### Initialize Model ###")
        self.init_and_load_model(prefer_best=False)

    def run(self):
        do_train = self.config.get("do_train", True)
        do_dev = self.config.get("do_dev", True)
        do_test = self.config.get("do_test", False)

        logger.info("### Loading Data ###")
        train_dataset = self.load_and_cache_examples(self.tokenizer, "train") if do_train else None
        dev_datasets = self.load_and_cache_examples(self.tokenizer, "dev") if do_dev else None
        test_datasets = self.load_and_cache_examples(self.tokenizer, "test") if do_test else None

        if do_train:
            logger.info("### Starting the training ###")
            self.train(train_dataset, dev_datasets)
        if do_test:
            logger.info("### Starting the testing with the best model ###")
            self.init_and_load_model(prefer_best=True)
            self.all_results["test_results"] = self.eval(test_datasets, is_test=True)
            with open(os.path.join(self.out_dir, "results.json"), "w") as f:
                json.dump(self.all_results, f, indent=4)


    def init_and_load_model(self, prefer_best):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.config["model"]["model"]]
        bert_model = model_class.from_pretrained(self.config["model"]["model_name"], cache_dir=self.config.get("transformer_cache_dir", None))
        if self.config["model"].get("version", "v2") == "v1":
            model = TransformerModel(self.config, bert_model)
        elif self.config["model"].get("version", "v2") == "v2":
            model = TransformerModelV2(self.config, bert_model)
        elif self.config["model"].get("version", "v2") == "v3":
            model = TransformerModelV3(self.config, bert_model)
        elif self.config["model"].get("version", "v2") == "softmax":
            model = TransformerModelSoftmax(self.config, bert_model)
        elif self.config["model"].get("version", "v2") == "mse":
            model = TransformerModelMSE(self.config, bert_model)
        elif self.config["model"].get("version", "v2") == "pairwise":
            model = TransformerModelPairwise(self.config, bert_model)
        best_model_file = os.path.join(self.out_dir, "best_model", "model.pty")
        latest_model_file = os.path.join(self.out_dir, "latest_model", "model.pty")
        if prefer_best and os.path.isfile(best_model_file):
            logger.info("Loading best model...")
            state_dict = torch.load(best_model_file)
            model.load_state_dict(state_dict)
        elif os.path.isfile(latest_model_file):
            logger.info("Loading latest model...")
            state_dict = torch.load(latest_model_file)
            model.load_state_dict(state_dict)
        elif "base_model" in self.config and os.path.isfile(self.config["base_model"]):
            logger.info("Loading a base model...")
            state_dict = torch.load(self.config["base_model"])
            #non_bert_keys = [key for key in state_dict.keys() if "bert" not in key]
            #to_delete = [key for key in non_bert_keys if "extend" in key]
            #if self.config.get("base_model_exclude_old_agents", False):
            #    to_delete.extend([key for key in non_bert_keys if "classifier" in key])
            #logger.info("Removing these parameters: {}".format(to_delete))
            #for key in to_delete:
            #    state_dict.pop(key, None)
            model.load_state_dict(state_dict, strict=False)
        if self.tokenizer is None:
            self.tokenizer = tokenizer_class.from_pretrained(self.config["model"]["model_name"], cache_dir=self.config.get("transformer_cache_dir", None))
        self.model = model
        self.model.to(self.device)

    def train(self, train_dataset, dev_datasets=None):
        self.model.train()
        train_config = self.config["train"]
        train_sampler = RandomSampler(train_dataset)
        gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=int(train_config["batch_size"]/gradient_accumulation_steps), collate_fn=self._collate)

        epochs = train_config["epochs"]

        if train_config.get("weight_constrain", False):
            param_copy = deepcopy(list(self.model.bert.parameters())+list(self.model.classifier.parameters())+list(self.model.adapter.parameters()))

        if train_config.get("max_steps", 0) > 0:
            t_total = train_config["max_steps"]
            epochs = train_config["max_steps"] // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * epochs

        if train_config.get("freeze_bert", False) or train_config.get("freeze_extend", False):
            logger.warning("! Freezing Bert Parameters !")
            for param in self.model.bert.parameters():
                param.requires_grad = False
            #if self.config["model"].get("version", "v2") == "v1":
            #    for param in self.model.preclass.parameters():
            #        param.requires_grad = False
        if train_config.get("freeze_extend", False):
            if self.config["model"].get("version", "v2") == "v3":
                for param in self.model.preclass1.parameters():
                    param.requires_grad = False
                for param in self.model.preclass2.parameters():
                    param.requires_grad = False
                self.model.embedding.requires_grad = False
            else:
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
                if self.config["model"].get("version", "v2") in ["v2", "softmax", "mse", "pairwise"]:
                    logger.warning("! Freezing Old Classifier Parameters !")
                    for param in self.model.adapter.parameters():
                        param.requires_grad = False


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = train_config.get("weight_decay", 0.0)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        learning_rate = train_config.get("learning_rate", 5e-5)
        adam_epsilon = train_config.get("adam_epsilon", 1e-8)
        warmup_fraction = train_config.get("warmup_fraction", 0.0)
        warmup_steps = t_total*warmup_fraction
        max_grad_norm = train_config.get("max_grad_norm", 1.0)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        latest_optim = os.path.join(self.out_dir, "latest_model", "optimizer.pty")
        latest_scheduler = os.path.join(self.out_dir, "latest_model", "scheduler.pty")
        if os.path.isfile(latest_optim) and os.path.isfile(latest_scheduler):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(latest_optim))
            scheduler.load_state_dict(torch.load(latest_scheduler))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Batchsize = %d", train_config["batch_size"])
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        epochs_trained = 0
        # Check if continuing training from a checkpoint
        if self.all_results["latest_epoch"] >= 0:
            # set global_step to global_step of last saved checkpoint from model path
            epochs_trained = self.all_results["latest_epoch"]+1

            logger.info("  Continuing training from checkpoint")
            logger.info("  Continuing training from epoch %d", epochs_trained)

        tr_loss, log_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs_trained, int(epochs), desc="Epoch", position=0)
        for current_epoch in train_iterator:
            train_dataset.resample()
            epoch_iterator = tqdm(train_dataloader, position=1, desc="Iteration")
            loss_log = tqdm(total=0, position=2, bar_format="{desc}")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2], "pos_weight": batch[3]}

                if train_config.get("loss_resample"):
                    outputs = self.model(**inputs, reduction="none")
                    loss = outputs[1]  # model outputs are always tuple in transformers (see doc)
                    if train_config["loss_resample_mode"] == "new":
                        resample_loss = loss[:, -1].cpu().tolist()
                    elif train_config["loss_resample_mode"] == "global":
                        resample_loss = torch.max(loss, dim=1)[0].cpu().tolist()
                    loss = torch.mean(loss)
                    train_dataset.add_losses(resample_loss)
                else:
                    outputs = self.model(**inputs)
                    loss = outputs[1]  # model outputs are always tuple in transformers (see doc)

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if max_grad_norm>0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    if train_config.get("weight_constrain", False):
                        params = list(self.model.bert.parameters())+list(self.model.classifier.parameters())+list(self.model.adapter.parameters())
                        constrain_loss = train_config["weight_constrain_factor"]*torch.stack([torch.sum((pc-p)**2) for pc, p in zip(param_copy, params)]).sum()
                        tr_loss = tr_loss + constrain_loss
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                if step%10 == 0:
                    l = (tr_loss-log_loss)/10
                    log_loss = tr_loss
                    loss_log.set_description_str("loss: {}".format(l))

            if dev_datasets is not None:
                logs = {}
                results = self.eval(dev_datasets, is_test=False)
                for key, value in results.items():
                    logs[key] = value
                results["epoch"] = current_epoch
                self.all_results["epoch_results"].append(results)

                # Save model checkpoint
                best_model_file = os.path.join(self.out_dir, "best_model", "model.pty")
                latest_model_file = os.path.join(self.out_dir, "latest_model", "model.pty")
                latest_optim = os.path.join(self.out_dir, "latest_model", "optimizer.pty")
                latest_scheduler = os.path.join(self.out_dir, "latest_model", "scheduler.pty")

                state_dict = self.model.state_dict()

                main_metric = self.config["dev"].get("main_metric", "accuracy")
                current_main_metric = results[main_metric]
                old_main_matric = self.all_results["epoch_results"][self.all_results["best_epoch"]][main_metric]
                # true iff result larger previous best and larger better or result smaller and larger not better
                better = (self.config["dev"].get("larger_is_better", True) ==
                          (current_main_metric > old_main_matric)) or current_epoch==0
                if better:
                    self.all_results["best_epoch"] = current_epoch
                    torch.save(state_dict, best_model_file)
                    logger.info("New best epoch result. Current epoch improves in {} from {:.4f} to {:.4f}".format(
                        main_metric, old_main_matric, current_main_metric))
                torch.save(state_dict, latest_model_file)
                logger.info("Saving latest model checkpoint")

                torch.save(optimizer.state_dict(), latest_optim)
                torch.save(scheduler.state_dict(), latest_scheduler)
                logger.info("Saving optimizer and scheduler states")
                self.all_results["latest_epoch"] = current_epoch

                with open(os.path.join(self.out_dir, "results.json"), "w") as f:
                    json.dump(self.all_results, f, indent=4)

    def eval(self, datasets, is_test=True):
        self.model.eval()
        if is_test:
            eval_config = self.config["test"]
        else:
            eval_config = self.config["dev"]

        results_all = defaultdict(lambda: list())
        confusion_matrix = [[0]*len(self.all_agents) for _ in self.all_agents]

        activation_mean = [[0]*len(self.all_agents) for _ in self.all_agents]
        activation_max = [[0]*len(self.all_agents) for _ in self.all_agents]
        activation_min = [[0]*len(self.all_agents) for _ in self.all_agents]
        activation_std = [[0]*len(self.all_agents) for _ in self.all_agents]

        for i, dataset in enumerate(datasets):
            agent_label = dataset[0][1]
            agent = self.all_agents[dataset[0][1]]
            dataloader = DataLoader(dataset, shuffle=False, batch_size=eval_config["batch_size"], collate_fn=self._collate)
            dataset_iterator = tqdm(dataloader, desc="Iteration ({})".format(agent))
            logger.info("Evaluating agent {}/{}: {}".format(i+1, len(datasets), agent))
            logger.info("{} questions".format(len(dataset)))
            start = datetime.datetime.now()
            ranks = []
            average_precisions = []
            activation = []
            for batch in dataset_iterator:
                batch = tuple(t.to(self.device) for t in batch[:2])
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs = self.model(**inputs)[0].cpu().numpy()
                for res in outputs:
                    rank = 0
                    precisions = 0
                    activation.append(res)
                    ranking = np.argsort(res)[::-1]
                    confusion_matrix[agent_label][ranking[0]] += 1
                    for j, answer in enumerate(ranking, start=1):
                        if answer == agent_label:
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
            #activation = (activation/float(len(ranks))).tolist()
            activation = np.array(activation)
            mean = np.mean(activation, axis=0).tolist()
            max = np.max(activation, axis=0).tolist()
            min = np.min(activation, axis=0).tolist()
            std = np.std(activation, axis=0).tolist()
            activation_mean[agent_label] = mean
            activation_min[agent_label] = min
            activation_max[agent_label] = max
            activation_std[agent_label] = std
            for key, value in results.items():
                results_all[key].append(value)
            results_all[agent] = results

        results_all["confusion_matrix"] = confusion_matrix
        results_all["activation"] = {}
        results_all["activation"]["mean"] = activation_mean
        results_all["activation"]["std"] = activation_std
        results_all["activation"]["min"] = activation_min
        results_all["activation"]["max"] = activation_max

        precision, recall, f1 = self._compute_f1(confusion_matrix)
        for dataset in datasets:
            agent = self.all_agents[dataset[0][1]]
            idx = self.all_agents.index(agent)
            results = results_all[agent]
            results["precision"] = precision[idx]
            results["recall"] = recall[idx]
            results["f1"] = f1[idx]
            results_all["precision"].append(precision[idx])
            results_all["recall"].append(recall[idx])
            results_all["f1"].append(f1[idx])

            logger.info('\nResults for agent {}'.format(agent))
            self._log_results(results)

        for key in ["accuracy", "precision", "recall", "f1", "mrr", "r@3", 'r@5', "query_per_second"]:
            results_all[key] = np.mean(results_all[key])
        logger.info('\nResults for all datasets:')
        self._log_results(results_all)

        if eval_config.get("print_confusion_matrix", False):
            self._log_confusion_matrix(confusion_matrix)

        return results_all

    def outlier_eval(self, outlier_agent):
        self.init_and_load_model(prefer_best=True)
        self.all_agents.append(outlier_agent)
        datasets = self.load_and_cache_examples(self.tokenizer, "test")

        self.model.eval()
        eval_config = self.config["test"]

        results_all = defaultdict(lambda: list())
        over_threshold5 = []
        over_threshold8 = []

        dataset = datasets[0]
        dataloader = DataLoader(dataset, shuffle=False, batch_size=eval_config["batch_size"], collate_fn=self._collate)
        dataset_iterator = tqdm(dataloader, desc="Iteration ({})".format(outlier_agent))
        logger.info("Outlier evaluation with agent: {}".format(outlier_agent))
        logger.info("{} questions".format(len(dataset)))
        for batch in dataset_iterator:
            batch = tuple(t.to(self.device) for t in batch[:2])
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            with torch.no_grad():
                outputs = self.model(**inputs)[0].cpu().numpy()
            for res in outputs:
                a = res.copy()
                a[a >= 0.5] = 1
                a[a < 0.5] = 0
                b = res.copy()
                b[b >= 0.8] = 1
                b[b < 0.8] = 0
                over_threshold5.append(a)
                over_threshold8.append(b)
        over_threshold5 = np.array(over_threshold5).mean(axis=0).tolist()
        over_threshold8 = np.array(over_threshold8).mean(axis=0).tolist()
        results_all = {"agent": outlier_agent, "over_threshold_0.5": over_threshold5, "over_threshold_0.8": over_threshold8}
        logger.info(results_all)
        self.all_results["outlier_results"] = results_all
        with open(os.path.join(self.out_dir, "results.json"), "w") as f:
            json.dump(self.all_results, f, indent=4)

    def outliers_eval(self):
        self.init_and_load_model(prefer_best=True)
        datasets = self.load_and_cache_examples(self.tokenizer, "test")

        self.model.eval()
        eval_config = self.config["test"]

        results_all = []

        for dataset in datasets:
            over_threshold5 = []
            over_threshold8 = []
            outlier_agent = self.all_agents[dataset[0][1]]
            dataloader = DataLoader(dataset, shuffle=False, batch_size=eval_config["batch_size"], collate_fn=self._collate)
            dataset_iterator = tqdm(dataloader, desc="Iteration ({})".format(outlier_agent))
            logger.info("Outlier evaluation with agent: {}".format(outlier_agent))
            logger.info("{} questions".format(len(dataset)))
            for batch in dataset_iterator:
                batch = tuple(t.to(self.device) for t in batch[:2])
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs = self.model(**inputs)[0].cpu().numpy()
                for res in outputs:
                    a = res.copy()
                    a[a >= 0.5] = 1
                    a[a < 0.5] = 0
                    b = res.copy()
                    b[b >= 0.8] = 1
                    b[b < 0.8] = 0
                    over_threshold5.append(a)
                    over_threshold8.append(b)
            over_threshold5 = np.array(over_threshold5).mean(axis=0).tolist()
            over_threshold8 = np.array(over_threshold8).mean(axis=0).tolist()
            results_all.append({"agent": outlier_agent, "over_threshold_0.5": over_threshold5, "over_threshold_0.8": over_threshold8})
        logger.info(results_all)
        self.all_results["outlier_results"] = results_all
        with open(os.path.join(self.out_dir, "results.json"), "w") as f:
            json.dump(self.all_results, f, indent=4)

    def activation_eval(self, new_agent):
        self.init_and_load_model(prefer_best=True)
        self.all_agents.append(new_agent)
        datasets = self.load_and_cache_examples(self.tokenizer, "test")

        self.model.eval()
        eval_config = self.config["test"]

        results_all = defaultdict(lambda: dict())

        for i, dataset in enumerate(datasets):
            agent = self.all_agents[dataset[0][1]]
            dataloader = DataLoader(dataset, shuffle=False, batch_size=eval_config["batch_size"], collate_fn=self._collate)
            dataset_iterator = tqdm(dataloader, desc="Iteration ({})".format(agent))
            logger.info("Evaluating agent {}/{}: {}".format(i+1, len(datasets), agent))
            logger.info("{} questions".format(len(dataset)))
            activation = []
            for batch in dataset_iterator:
                batch = tuple(t.to(self.device) for t in batch[:2])
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs = self.model(**inputs)[0].cpu().numpy()
                for res in outputs:
                    top_activation = np.sort(res)[-1].item()
                    activation.append(top_activation)
            if agent == new_agent:
                results_all["outlier"][agent] = activation
            else:
                results_all["trained"][agent] = activation
        with open(os.path.join(self.out_dir, "results_activation.json"), "w") as f:
            json.dump(results_all, f, indent=4)

    def _log_results(self, results):
        logger.info('Accuracy: {}'.format(results['accuracy']))
        logger.info('Precision: {}'.format(results['precision']))
        logger.info('Recall: {}'.format(results['recall']))
        logger.info('F1: {}'.format(results['f1']))
        logger.info('MRR: {}'.format(results['mrr']))
        logger.info('R@3: {}'.format(results['r@3']))
        logger.info('R@5: {}'.format(results['r@5']))
        logger.info('Queries/second: {}'.format(results['query_per_second']))

    def _log_confusion_matrix(self, confusion_matrix):
        cell_width = max([5]+[len(s) for s in self.all_agents])
        logger.info("Confusion Matrix:")
        logger.info("|".join(["".ljust(cell_width)]+[str(s).ljust(cell_width) for s in self.all_agents]))
        for agent, row in zip(self.all_agents, confusion_matrix):
            logger.info("|".join([agent.ljust(cell_width)]+[str(s).ljust(cell_width) for s in row]))

    def _compute_f1(self, confusion_matrix):
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


    def _collate(self, samples):
        input_ids, labels = zip(*samples)
        max_len = min(self.config["model"]["max_length"], max([len(input) for input in input_ids]))
        attention_mask = [[1]*len(input)+[0]*(max_len-len(input)) for input in input_ids]
        input_ids = [input+[0]*(max_len-len(input)) for input in input_ids]
        if self.config["model"].get("version", "v2") == "softmax":
            pos_weights = torch.FloatTensor(self.agent_weights_ce)
            one_hot_labels = torch.LongTensor(labels)  # not really one hot
        elif self.config["model"].get("version", "v2") == "pairwise":
            pos_weights = torch.FloatTensor(self.agent_weights_ce)
            one_hot_labels = torch.LongTensor(labels)  # not really one hot
            loss_label = torch.zeros((len(labels), len(self.all_agents))).long() - 1
            loss_label[:,0] = one_hot_labels
            one_hot_labels = loss_label
        else:
            pos_weights = torch.FloatTensor(self.agent_weights)
            one_hot_labels = torch.FloatTensor(len(labels), len(self.config["all_agents"])) \
                .zero_() \
                .scatter_(1, torch.Tensor(labels).long().unsqueeze(1), 1)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, attention_mask, one_hot_labels, pos_weights

    def load_and_cache_examples(self, tokenizer, partition):
        directory = os.path.join(self.config["cache_dir"], self.config["model"]["model_name"].replace("/", "_"), partition)
        os.makedirs(directory, exist_ok=True)
        base_paths = self.config[partition]["base_path"]
        all_examples = {"input_ids": [], "label": []}
        num_examples_per_agent = {}
        truncate_idx = 0 # for multiple truncate values; looping through
        extend_truncate_idx = 0
        for i, (agent, file_names) in enumerate(self.config[partition]["agents"].items()):
            file_path = os.path.join(directory, "{}.pyt".format(agent))
            if os.path.exists(file_path):
                logger.info("Loading cached {} set for {}".format(partition, agent))
                examples = torch.load(file_path)
            else:
                logger.info("No cached data for {}. Creating it...".format(agent))
                data = _read_file(file_names, base_paths)
                res = tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=self.config["model"]["max_length"],
                                                  return_token_type_ids=False, return_attention_masks=False)
                examples = {"input_ids": res["input_ids"]}
                torch.save(examples, file_path)
                logger.info("Cached {} set for {} data to {}".format(partition, agent, file_path))
            idx = self.all_agents.index(agent)
            examples["label"] = [idx]*len(examples["input_ids"])
            truncate = self.config[partition].get("truncate", len(examples["input_ids"]))
            if isinstance(truncate, list):
                truncate = truncate[truncate_idx]
            true_truncate = truncate # for epoch resample
            if self.config.get("agents_extended", 0)>0 and \
                    agent in self.all_agents[-self.config.get("agents_extended", 0):]:
                truncate = self.config[partition].get("extend_truncate", truncate)
                if isinstance(truncate, list):
                    truncate = truncate[extend_truncate_idx]
                    extend_truncate_idx = (extend_truncate_idx+1)%len(self.config[partition].get("extend_truncate", truncate))
                true_truncate = truncate
            elif isinstance(self.config[partition].get("truncate", len(examples["input_ids"])), list):
                truncate_idx = (truncate_idx+1)%len(self.config[partition].get("truncate", len(examples["input_ids"])))
            if self.config["train"].get("epoch_resample", False) or self.config["train"].get("loss_resample", False):
                truncate = self.config[partition].get("extend_truncate", truncate)
                if isinstance(truncate, list):
                    truncate = max(truncate)
            chosen_idxs = np.random.choice(len(examples["input_ids"]), truncate)
            if self.config.get("agents_extended", 0)>0 and \
                    agent in self.all_agents[-self.config.get("agents_extended", 0):] and \
                    (self.config["train"].get("epoch_resample", False) or self.config["train"].get("loss_resample", False)) :
                extend_examples = {"input_ids": [], "label": []}
                extend_examples["input_ids"] = [examples["input_ids"][idx] for idx in chosen_idxs]
                extend_examples["label"] = [examples["label"][idx] for idx in chosen_idxs]
            else:
                all_examples["input_ids"].append([examples["input_ids"][idx] for idx in chosen_idxs])
                all_examples["label"].append([examples["label"][idx] for idx in chosen_idxs])
            num_examples_per_agent[agent] = len(examples["label"][:true_truncate])  # can be smaller than truncate
            logger.info("{} samples for {} {}".format(len(examples["label"][:true_truncate]), partition, agent))

        if partition == "train":
            total_examples = sum(num_examples_per_agent.values())
            for i, agent in enumerate(self.all_agents):
                if agent in num_examples_per_agent:
                    # case with all agents equal number of examples: positive weight = #agents-1
                    self.agent_weights[i] = (total_examples-num_examples_per_agent[agent])/num_examples_per_agent[agent]
                    self.agent_weights_ce[i] = (total_examples/len(self.all_agents))/num_examples_per_agent[agent]
            if self.config["train"].get("epoch_resample", False):
                dataset = CustomResampleDataset(input_ids=all_examples["input_ids"], labels=all_examples["label"],
                                                truncate=self.config["train"]["truncate"],
                                                extend_input_ids=extend_examples["input_ids"], extend_labels=extend_examples["label"])
            elif self.config["train"].get("loss_resample", False):
                dataset = CustomLossResampleDataset(input_ids=all_examples["input_ids"], labels=all_examples["label"],
                                                    truncate=self.config["train"]["truncate"],
                                                    extend_input_ids=extend_examples["input_ids"], extend_labels=extend_examples["label"])
            else:
                dataset = CustomDataset([example for examples in all_examples["input_ids"] for example in examples],
                                        [label for labels in all_examples["label"] for label in labels])
        else:
            dataset = [CustomDataset(examples,  labels)
                       for examples, labels in zip(all_examples["input_ids"], all_examples["label"])]
        return dataset

    def load_examples_with_text(self, tokenizer, partition):
        directory = os.path.join(self.config["cache_dir"], self.config["model"]["model_name"].replace("/", "_"), partition)
        base_paths = self.config[partition]["base_path"]
        all_examples = {"input_ids": [], "label": []}
        all_data = []
        num_examples_per_agent = {}
        truncate_idx = 0 # for multiple truncate values; looping through
        extend_truncate_idx = 0
        for i, (agent, file_names) in enumerate(self.config[partition]["agents"].items()):
            file_path = os.path.join(directory, "{}.pyt".format(agent))
            logger.info("No cached data for {}. Creating it...".format(agent))
            data = _read_file(file_names, base_paths)
            res = tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=self.config["model"]["max_length"],
                                              return_token_type_ids=False, return_attention_masks=False)
            examples = {"input_ids": res["input_ids"]}
            idx = self.all_agents.index(agent)
            examples["label"] = [idx]*len(examples["input_ids"])
            truncate = self.config[partition].get("truncate", len(examples["input_ids"]))
            if isinstance(truncate, list):
                truncate = truncate[truncate_idx]
            true_truncate = truncate # for epoch resample
            if self.config.get("agents_extended", 0)>0 and \
                    agent in self.all_agents[-self.config.get("agents_extended", 0):]:
                truncate = self.config[partition].get("extend_truncate", truncate)
                if isinstance(truncate, list):
                    truncate = truncate[extend_truncate_idx]
                    extend_truncate_idx = (extend_truncate_idx+1)%len(self.config[partition].get("extend_truncate", truncate))
                true_truncate = truncate
            elif isinstance(self.config[partition].get("truncate", len(examples["input_ids"])), list):
                truncate_idx = (truncate_idx+1)%len(self.config[partition].get("truncate", len(examples["input_ids"])))
            if self.config["train"].get("epoch_resample", False) or self.config["train"].get("loss_resample", False):
                truncate = self.config[partition].get("extend_truncate", truncate)
                if isinstance(truncate, list):
                    truncate = max(truncate)
            chosen_idxs = np.random.choice(len(examples["input_ids"]), truncate)
            if self.config.get("agents_extended", 0)>0 and \
                    agent in self.all_agents[-self.config.get("agents_extended", 0):] and \
                    (self.config["train"].get("epoch_resample", False) or self.config["train"].get("loss_resample", False)) :
                extend_examples = {"input_ids": [], "label": []}
                extend_examples["input_ids"] = [examples["input_ids"][idx] for idx in chosen_idxs]
                extend_examples["label"] = [examples["label"][idx] for idx in chosen_idxs]
            else:
                all_examples["input_ids"].append([examples["input_ids"][idx] for idx in chosen_idxs])
                all_examples["label"].append([examples["label"][idx] for idx in chosen_idxs])
            all_data.append([data[idx] for idx in chosen_idxs])

        dataset = [CustomDataset(examples,  labels)
                   for examples, labels in zip(all_examples["input_ids"], all_examples["label"])]
        return dataset, all_data

    def eval_for_manual_annotation(self, datasets, questions, chose=50):
        import pandas as pd
        self.model.eval()
        eval_config = self.config["test"]

        errors = []
        for i, (dataset, qs) in enumerate(zip(datasets, questions)):
            agent_label = dataset[0][1]
            agent = self.all_agents[dataset[0][1]]
            dataloader = DataLoader(dataset, shuffle=False, batch_size=eval_config["batch_size"], collate_fn=self._collate)
            dataset_iterator = tqdm(dataloader, desc="Iteration ({})".format(agent))
            logger.info("Evaluating agent {}/{}: {}".format(i+1, len(datasets), agent))
            logger.info("{} questions".format(len(dataset)))
            activation = []
            for i, batch in enumerate(dataset_iterator):
                batch = tuple(t.to(self.device) for t in batch[:2])
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs = self.model(**inputs)[0].cpu().numpy()
                for j, res in enumerate(outputs):
                    rank1 = np.argsort(res)[::-1][0]
                    if rank1 != agent_label:
                        errors.append((qs[i*eval_config["batch_size"]+j], agent, self.all_agents[rank1]))
        chosen_idx = np.random.choice(len(errors), chose, replace=False)
        errors = [errors[i] for i in chosen_idx]
        key = []
        shuffle = []
        for e in errors:
            a, b = np.random.choice(2, 2, replace=False)
            shuffle.append((e[0], e[1+a], e[1+b]))
            key.append(a)
        df = pd.DataFrame(shuffle, columns=["Question", "A", "B"])
        key = pd.DataFrame(key, columns=["Key"])
        return df, key


def _read_file(paths, base_paths):
    if not isinstance(paths, list):
        paths = [paths]
    if not isinstance(base_paths, list):
        base_paths = [base_paths]
    data = []
    for path in paths:
        exists = False
        for base_path in base_paths:
            entire_path = os.path.join(base_path, path)
            if os.path.isfile(entire_path):
                exists = True
                with open(entire_path, "r", encoding="utf-8") as f:
                    for l in f.readlines():
                        l = l.replace("?", "").lower().strip()
                        if l == "":
                            continue
                        data.append(l)
        if not exists:
            raise FileNotFoundError("{} was not found in any base path".format(path))
    return data

class CustomDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, index: int):
        return self.input_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def resample(self):
        pass

class CustomResampleDataset(Dataset):
    def __init__(self, input_ids, labels, truncate, extend_input_ids, extend_labels):
        self.input_ids = []
        self.labels = []
        self.other_ids = input_ids
        self.other_labels = labels
        self.truncate = truncate
        self.extend_ids = extend_input_ids
        self.extend_labels = extend_labels
        self.resample()

    def __getitem__(self, index: int):
        return self.input_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def resample(self):
        self.input_ids = []
        self.labels = []
        for ids, labels in zip(self.other_ids, self.other_labels):
            chosen_idxs = np.random.choice(len(ids), self.truncate)
            self.input_ids.extend([ids[idx] for idx in chosen_idxs])
            self.labels.extend([labels[idx] for idx in chosen_idxs])
        self.input_ids.extend(self.extend_ids)
        self.labels.extend(self.extend_labels)

class CustomLossResampleDataset(Dataset):
    def __init__(self, input_ids, labels, truncate, extend_input_ids, extend_labels):
        self.input_ids = []
        self.labels = []
        self.other_ids = input_ids
        self.other_labels = labels
        self.truncate = truncate
        self.extend_ids = extend_input_ids
        self.extend_labels = extend_labels
        for ids, labels in zip(self.other_ids, self.other_labels):
            chosen_idxs = np.random.choice(len(ids), self.truncate)
            self.input_ids.extend([ids[idx] for idx in chosen_idxs])
            self.labels.extend([labels[idx] for idx in chosen_idxs])
        self.input_ids.extend(self.extend_ids)
        self.labels.extend(self.extend_labels)
        self.get_order = []
        self.get_losses = []

    def __getitem__(self, index: int):
        self.get_order.append(index)
        return self.input_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def add_losses(self, losses):
        self.get_losses.extend(losses)

    def resample(self):
        cutoff_idx = len(self.labels)-len(self.extend_labels)
        get_order, get_losses = [], []
        for idx, loss in zip(self.get_order, self.get_losses):
            if idx < cutoff_idx:
                get_order.append(idx)
                get_losses.append(loss)
        self.get_order, self.get_losses = [], []
        argpartition = np.argpartition(get_losses, 3*len(get_losses)//4)[:3*len(get_losses)//4] # 75% lowest loss

        resample_idxs = [[] for _ in range(len(self.other_ids))]
        for argidx in argpartition:
            resample_idxs[get_order[argidx]//self.truncate].append(get_order[argidx])

        for ids, labels, resample_idx in zip(self.other_ids, self.other_labels, resample_idxs):
            if len(resample_idx) > 0:
                chosen_idxs = np.random.choice(len(ids), len(resample_idx))
                for ridx, cidx in zip(resample_idx, chosen_idxs):
                    self.input_ids[ridx] = ids[cidx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml")
    args = parser.parse_args()
    config = yaml.load(open(args.config))
    manager = Manager(config)
    manager.run()
