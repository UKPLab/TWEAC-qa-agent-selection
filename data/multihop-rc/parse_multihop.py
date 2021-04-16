import json
import os

def hotpot(folder, out_folder):
    train = json.load(open(os.path.join(folder, "hotpot_train_v1.1.json")))
    test = json.load(open(os.path.join(folder,  "hotpot_dev_distractor_v1.json")))

    train_q = []
    test_q = []

    for q in train:
        train_q.append(q["question"])
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for q in test:
        test_q.append(q["question"])

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "hotpot.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    hotpot(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\multihop-rc\hotpot", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\multihop-rc")
