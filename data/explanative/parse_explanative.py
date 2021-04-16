import pandas as pd
from collections import OrderedDict
import os
def wikiqa(folder, out_folder):
    train_dt = pd.read_csv(os.path.join(folder, "WikiQA-train.tsv"), encoding="utf-8", sep="\t")
    dev_dt = pd.read_csv(os.path.join(folder, "WikiQA-dev.tsv"), encoding="utf-8", sep="\t")
    test_dt = pd.read_csv(os.path.join(folder, "WikiQA-test.tsv"), encoding="utf-8", sep="\t")

    train_q = train_dt["Question"].unique()
    dev_q = dev_dt["Question"].unique()
    test_q = test_dt["Question"].unique()

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "wikiqa.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")

def wikipassageqa(folder, out_folder):
    train_dt = pd.read_csv(os.path.join(folder, "train.tsv"), encoding="utf-8", sep="\t")
    dev_dt = pd.read_csv(os.path.join(folder, "dev.tsv"), encoding="utf-8", sep="\t")
    test_dt = pd.read_csv(os.path.join(folder, "test.tsv"), encoding="utf-8", sep="\t")

    train_q = train_dt["Question"].unique()
    dev_q = dev_dt["Question"].unique()
    test_q = test_dt["Question"].unique()

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "wikipassageqa.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")



if __name__ == "__main__":
    wikiqa(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\explanative\WikiQACorpus", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\explanative")
    wikipassageqa(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\explanative\WikiPassageQA", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\explanative")