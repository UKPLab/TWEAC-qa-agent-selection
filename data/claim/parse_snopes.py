import pandas as pd
from collections import OrderedDict
import os
def snopes(file, out_folder):
    dt = pd.read_csv(file, encoding="utf-8")
    claims = OrderedDict()
    for c in dt["Claim"]:
        claims[c] = None
    claims = list(claims.keys())

    split_idx1 = int(len(claims)*0.6)
    split_idx2 = int(len(claims)*0.8)
    train_q = claims[:split_idx1]
    dev_q = claims[split_idx1: split_idx2]
    test_q = claims[split_idx2:]

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "snopes.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    snopes(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\claim\Snopes_AnH_2019\SnopesCorpus14June2017_evidence_v2.csv", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\claim")
