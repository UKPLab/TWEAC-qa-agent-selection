from run_transformer import Manager
import argparse
import yaml
from copy import deepcopy
import logging
import os
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def finetune(args):
    source_config = yaml.load(open(args.fconfig))
    skip = args.skip.split(",")
    for i, agent in enumerate(source_config["all_agents"]):
        if agent in skip:
            logger.info("Skipping agent {}".format(agent))
            continue
        config = deepcopy(source_config)
        config["folder_name_prefix"] = agent
        config["all_agents"].remove(agent)
        config["train"]["agents"].pop(agent)
        config["dev"]["agents"].pop(agent)
        if "test" in config:
            config["test"]["agents"].pop(agent)
        logger.info("########################################### ")
        logger.info("Training {}/{}; agent left out: {}".format(i+1, len(source_config["all_agents"]), agent))
        logger.info("########################################### ")

        manager = Manager(config)
        manager.run()


def extend(args):
    source_config = yaml.load(open(args.econfig))
    skip = args.skip.split(",")
    finetune_folder = source_config["base_model"]
    folders = os.listdir(finetune_folder)
    for i, agent in enumerate(source_config["all_agents"]):
        logger.info("########################################### ")
        logger.info("Training {}/{}; agent extended: {}".format(i+1, len(source_config["all_agents"]), agent))
        logger.info("########################################### ")
        if agent in skip:
            logger.info("Skipping agent {}".format(agent))
            continue
        finetuned_folder = [f for f in folders if agent in f and "testset" not in f][0]
        base_path = os.path.join(finetune_folder, finetuned_folder, "best_model", "model.pty")
        config = deepcopy(source_config)
        config["base_model"] = base_path
        config["folder_name_prefix"] = agent
        config["all_agents"].remove(agent)
        config["all_agents"].append(agent) # changing order

        manager = Manager(config)
        manager.run()

# def outlier(args):
#     source_config = yaml.load(open(args.oconfig))
#     skip = args.skip.split(",")
#     finetune_folder = source_config["base_model"]
#     folders = os.listdir(finetune_folder)
#     for i, agent in enumerate(source_config["all_agents"]):
#         logger.info("########################################### ")
#         logger.info("Training {}/{}; agent outlier: {}".format(i+1, len(source_config["all_agents"]), agent))
#         logger.info("########################################### ")
#         if agent in skip:
#             logger.info("Skipping agent {}".format(agent))
#             continue
#         finetuned_folder = [f for f in folders if agent in f][0]
#         base_path = os.path.join(finetune_folder, finetuned_folder, "best_model", "model.pty")
#         config = deepcopy(source_config)
#         config["base_model"] = base_path
#         config["folder_name_prefix"] = agent
#         config["all_agents"].remove(agent)
#         manager = Manager(config)
#         manager.activation_eval(agent)

# def retrain(args):
#     source_config = yaml.load(open(args.rconfig))
#     skip = args.skip.split(",")
#     finetune_folder = source_config["base_model"]
#     folders = os.listdir(finetune_folder)
#     for i, agent in enumerate(source_config["all_agents"]):
#         logger.info("########################################### ")
#         logger.info("Training {}/{}; agent extended: {}".format(i+1, len(source_config["all_agents"]), agent))
#         logger.info("########################################### ")
#         if agent in skip:
#             logger.info("Skipping agent {}".format(agent))
#             continue
#         finetuned_folder = [f for f in folders if agent in f][0]
#         base_path = os.path.join(finetune_folder, finetuned_folder, "best_model", "model.pty")
#         config = deepcopy(source_config)
#         config["base_model"] = base_path
#         config["folder_name_prefix"] = agent
#         config["all_agents"].remove(agent)
#         config["train"]["agents"].pop(agent)
#         config["dev"]["agents"].pop(agent)
#         manager = Manager(config)
#         manager.run()
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--econfig", default="config.yaml")
    parser.add_argument("--fconfig", default="config.yaml")
    # parser.add_argument("--rconfig", default="config.yaml")
    # parser.add_argument("--oconfig", default="config.yaml")
    parser.add_argument("--skip", default="")
    parser.add_argument("--modes", default="")
    args = parser.parse_args()
    if "finetune" in args.modes:
        finetune(args)
    # if "retrain" in args.modes:
    #     retrain(args)
    if "extend" in args.modes:
        extend(args)
    # if "outlier" in args.modes:
    #     outlier(args)