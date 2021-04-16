import json
import os

def nlu(folder, out_folder, intent, out_name):
    train = json.load(open(os.path.join(folder, "train_"+intent+".json")))[intent]
    test = json.load(open(os.path.join(folder, "validate_"+intent+".json")))[intent]

    train_q = []
    test_q = []

    for q in train:
        entire_q = ""
        for chunk in q["data"]:
            entire_q += chunk["text"]
        train_q.append(entire_q)
    split_idx = int(len(train_q)*0.75)
    dev_q = train_q[split_idx:]
    train_q = train_q[:split_idx]

    for q in test:
        entire_q = ""
        for chunk in q["data"]:
            entire_q += chunk["text"]
        test_q.append(entire_q)

    for qs, split in zip([train_q, dev_q, test_q], ["train", "dev", "test"]):
        os.makedirs(os.path.join(out_folder, split), exist_ok=True)
        with open(os.path.join(out_folder, split, out_name), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q+"\n")


if __name__ == "__main__":
    nlu(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\intents\movie", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\intents", "SearchScreeningEvent", "movie_schedule.txt")
    nlu(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\intents\weather", r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\intents", "GetWeather", "weather.txt")
