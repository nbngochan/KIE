NODE_CLASS_NAMES = [
    "company",
    "address",
    "date",
    "total",
    "other"
]
N_NODE_CLASSES = len(NODE_CLASS_NAMES)
NODE_LABEL_MAP = {item: idx for idx, item in enumerate(NODE_CLASS_NAMES)}
BERT_FEAT_SIZE = 768