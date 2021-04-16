import os

def classes_to_list(train_file):
    corse_classes = set()
    fine_classes = set()

    with open(train_file) as f:
        for line in f.readlines():
            label = line.split()[0]
            corse_label = label.split(":")[0]
            corse_classes.add(corse_label)
            fine_classes.add(label)

    print(", ".join("{}".format(c) for c in corse_classes))
    print(", ".join("{}".format(c) for c in fine_classes))

CORSE_CLASSES = ['HUM', 'ENTY', 'ABBR', 'DESC', 'LOC', 'NUM']
FINE_CLASSES = ['ENTY:sport', 'HUM:gr', 'NUM:weight', 'ENTY:body', 'HUM:desc', 'ENTY:letter', 'LOC:city', 'LOC:country', 'NUM:speed', 'ENTY:lang', 'ENTY:product', 'ENTY:word', 'NUM:volsize', 'NUM:money', 'NUM:date', 'ENTY:termeq', 'ENTY:event', 'NUM:ord', 'ENTY:veh', 'LOC:mount', 'ENTY:food', 'ENTY:color', 'NUM:dist', 'ENTY:animal', 'ENTY:substance', 'ENTY:symbol', 'NUM:other', 'DESC:manner', 'HUM:ind', 'LOC:state', 'NUM:count', 'ABBR:abb', 'NUM:perc', 'DESC:def', 'ENTY:dismed', 'ENTY:techmeth', 'LOC:other', 'NUM:period', 'ENTY:instru', 'NUM:code', 'ABBR:exp', 'NUM:temp', 'ENTY:other', 'DESC:desc', 'ENTY:plant', 'ENTY:currency', 'HUM:title', 'ENTY:cremat', 'ENTY:religion', 'DESC:reason']

def split_qc(folder):
    train = []
    dev = []
    dev_count = [0] * len(FINE_CLASSES)
    with open(os.path.join(folder, "full_5500.label")) as f:
        for line in f.readlines():
            label = line.split()[0]
            idx = FINE_CLASSES.index(label)
            if dev_count[idx] < 10:
                dev.append(line)
                dev_count[idx] += 1
            else:
                train.append(line)
    for qs, name in zip([train, dev], ["train_5000.label", "dev_500.label"]):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q)


if __name__ == "__main__":
    #classes_to_list(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\question_classification\train_5500.label")
    split_qc(r"C:\Users\Gregor\Documents\Programming\square-skill-selector\data\question_classification")