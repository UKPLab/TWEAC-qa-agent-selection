import json
import os

def squad(folder, out_folder):
    train = json.load(open(os.path.join(folder, "train-v2.0.json")))["data"]
    test = json.load(open(os.path.join(folder,  "dev-v2.0.json")))["data"]

    train_q = []
    test_q = []

    for data in train:
        for paragraph in data["paragraphs"]:
            for q in paragraph["qas"]:
                train_q.append(q["question"])
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for data in test:
        for paragraph in data["paragraphs"]:
            for q in paragraph["qas"]:
                test_q.append(q["question"])

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "squad.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    squad(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\span-rc\squad", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\span-rc")
