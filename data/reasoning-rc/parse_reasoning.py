import json
import os

def drop(folder, out_folder):
    train = json.load(open(os.path.join(folder, "drop_dataset_train.json")))
    test = json.load(open(os.path.join(folder,  "drop_dataset_dev.json")))

    train_q = []
    test_q = []

    for key in train:
        for qa_pair in train[key]["qa_pairs"]:
            train_q.append(qa_pair["question"])
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for key in test:
        for qa_pair in test[key]["qa_pairs"]:
            test_q.append(qa_pair["question"])

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "drop.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    drop(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\reasoning-rc\drop_dataset", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\reasoning-rc")
