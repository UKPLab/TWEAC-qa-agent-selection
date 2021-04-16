import json
import os

def qald(in_folder, out_folder):
    train = json.load(open(os.path.join(in_folder, "qald-7-train-en-wikidata.json")))
    test = json.load(open(os.path.join(in_folder, "qald-7-test-en-wikidata-withoutanswers.json")))

    train_q = []
    test_q = []

    for qs in train["questions"]:
        for q in qs["question"]:
            train_q.append(q["string"])
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for qs in test["questions"]:
        for q in qs["question"]:
            test_q.append(q["string"])

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "qald-7.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")

def websqp(in_folder, out_folder):
    train = json.load(open(os.path.join(in_folder, "WebQSP.train.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(in_folder, "WebQSP.test.json"), encoding="utf-8"))

    train_q = []
    test_q = []

    for q in train["Questions"]:
        train_q.append(q["RawQuestion"])
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for q in test["Questions"]:
        test_q.append(q["RawQuestion"])

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, "webqsp.txt"), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    qald(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\kbqa\qald", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\kbqa")
    websqp(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\kbqa\WebQSP\data", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\kbqa")