import json
import os

def boolq(folder, out_folder):
    train = [json.loads(line) for line in open(os.path.join(folder, "train.jsonl"), encoding="utf-8").readlines()]
    test = [json.loads(line) for line in open(os.path.join(folder, "dev.jsonl"), encoding="utf-8").readlines()]

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
        with open(os.path.join(out_folder, split, "boolq.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    boolq(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\bool-rc\boolq", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\bool-rc")
