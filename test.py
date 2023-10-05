import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.NODE_CLASS import N_NODE_CLASSES, NODE_CLASS_NAMES
from graph_modules.graph_dataset import GraphDataset
from graph_modules.node_classifier import NodeClassifier
from graph_modules.utils import ConfigReader, calculate_confusion_matrix


def main(args):
    # Read configuration
    config = ConfigReader(args.json_config)

    if args.store_path is None:
        args.store_path = './store'

    if not os.path.isdir(args.store_path):
        os.mkdir(args.store_path)

    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    # validation data loader
    test_dataset = GraphDataset(data_path=args.test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 collate_fn=test_dataset.arrange_batch)

    model = NodeClassifier(config.model)
    model.to(device)

    n_classes = model.n_classes
    class_names = model.class_names

    # Check and load pretrained model
    assert os.path.exists(args.trained), 'No trained model specifying ...'
    if args.trained is not None and args.trained != 'None':
        model.load_state_dict(torch.load(args.trained, map_location=device))
        print("Restored from {}".format(args.trained))

    model.eval()

    # metrics
    node_instances = 0
    correct_nodes = 0

    node_predictions = []
    node_groundtruths = []
    for graph, node_labels, _ in tqdm(test_dataloader):
        node_labels = torch.tensor(node_labels, dtype=torch.long).to(device)
        graph = graph.to(device)

        node_logits = model(graph)

        _, node_preds = torch.max(node_logits, dim=1)
        correct_nodes += torch.sum(node_preds == node_labels)
        node_instances += len(node_preds)

        node_predictions.extend(node_preds.to("cpu").numpy())
        node_groundtruths.extend(node_labels.to("cpu").numpy())

    accuracy = float(correct_nodes) / node_instances
    print("Node Acc {:.4f} ".format(accuracy))

    node_f1_score, main_class_f1_score = calculate_confusion_matrix(node_predictions, node_groundtruths,
                                                                    n_classes, class_names, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification on graph-based documents.')
    parser.add_argument("--test_dataset", required=True, help="path to testing data set")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--trained", default=None, help="path to pre-trained model")
    parser.add_argument("--store_path", default=None, help="path to save trained model")
    parser.add_argument("--json_config", required=True, help="path to json file of configuration")
    args = parser.parse_args()
    print(args)
    main(args)
