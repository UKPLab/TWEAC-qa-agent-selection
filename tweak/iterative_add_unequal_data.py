from run_transformer import Manager
import argparse
import yaml
from copy import deepcopy
import torch
import logging
import os
import numpy as np
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extend():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml")
    parser.add_argument("--data_config", default="config.yaml")
    parser.add_argument("--start", default=0)
    parser.add_argument("--stop", default=0)
    args = parser.parse_args()
    source_config = yaml.load(open(args.config))
    data_config = yaml.load(open(args.data_config))
    extend_agents = data_config["all_agents"]
    stop = int(args.stop)
    if stop == 0:
        stop = len(extend_agents)
    start = int(args.start)
    base_model = source_config["base_model"]

    if "shuffle_seed" in source_config:
        np.random.seed(source_config["shuffle_seed"])
        extend_agents = np.random.permutation(extend_agents)

    extend_agents = extend_agents[0: stop]
    for i, agent in enumerate(extend_agents):
        logger.info("########################################### ")
        logger.info("Training {}/{}; agent extended: {}".format(i+1, len(extend_agents), agent))
        logger.info("########################################### ")

        source_config["train"]["agents"][agent] = data_config["train"]["agents"][agent]
        source_config["dev"]["agents"][agent] = data_config["dev"]["agents"][agent]
        source_config["all_agents"].append(agent)

        if i<start:
            logger.info("Skipping agent {}".format(agent))
            continue

        config = deepcopy(source_config)
        config["base_model"] = base_model
        config["folder_name_prefix"] = str(i)+agent
        if "sampling" in config:
            config["train"]["truncate"] = max(1, max(config["train"]["extend_truncate"])//(len(config["all_agents"])-1))
            logger.info("Sampling {} examples from each old agent".format(config["train"]["truncate"]))
        if isinstance(config["train"]["extend_truncate"], list):
            config["train"]["extend_truncate"] = [config["train"]["extend_truncate"][i%len(config["train"]["extend_truncate"])],
                                                  max(config["train"]["extend_truncate"])]
        config["disable_hash_in_folder_name"] = True
        manager = Manager(config)
        manager.run()
        folder_name = "{}_{}_{}_".format(config["folder_name_prefix"], config["model"]["model"], config["model"]["model_name"])
        base_model = os.path.join(config["out_dir"], folder_name, "best_model", "model.pty")
        merge_extended(base_model)


def merge_extended(state_dict_file):
    state_dict = torch.load(state_dict_file)
    extend_adapter_weight = state_dict.pop("extend_adapter.weight")
    extend_adapter_bias = state_dict.pop("extend_adapter.bias")
    extend_classifier_weight = state_dict.pop("extend_classifier.weight")
    extend_classifier_bias = state_dict.pop("extend_classifier.bias")
    state_dict["adapter.weight"] = torch.cat((state_dict["adapter.weight"], extend_adapter_weight), 0)
    state_dict["adapter.bias"] = torch.cat((state_dict["adapter.bias"], extend_adapter_bias), 0)
    state_dict["classifier.weight"] = torch.cat((state_dict["classifier.weight"], extend_classifier_weight), 0)
    state_dict["classifier.bias"] = torch.cat((state_dict["classifier.bias"], extend_classifier_bias), 0)
    torch.save(state_dict, state_dict_file)


if __name__ == "__main__":
    extend()
