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

from config_graph.NODE_CLASS import N_NODE_CLASSES, NODE_CLASS_NAMES
from graph_modules.graph_dataset import GraphDataset
from graph_modules.node_classifier import NodeClassifier
from graph_modules.utils import ConfigReader, calculate_confusion_matrix


def validate(model, epoch, data_loader, criterion, summary_writer, device):
    model.eval()
    # Check activated modules
    # metrics
    node_loss = []
    node_instances = 0
    correct_nodes = 0

    node_predictions = []
    node_groundtruths = []
    for graph, node_labels, _ in tqdm(data_loader):
        node_labels = torch.tensor(node_labels, dtype=torch.long).to(device)
        graph = graph.to(device)

        node_logits = model(graph)

        n_loss = criterion(node_logits, node_labels)
        node_loss.append(n_loss.item())

        _, node_preds = torch.max(node_logits, dim=1)
        correct_nodes += torch.sum(node_preds == node_labels)
        node_instances += len(node_preds)

        node_predictions.extend(node_preds.to("cpu").numpy())
        node_groundtruths.extend(node_labels.to("cpu").numpy())

    total_loss = np.mean(np.array(node_loss))
    accuracy = float(correct_nodes) / node_instances
    print("Evaluation | Loss {:.4f}".format(total_loss))
    print("Node Loss {:.4f} | Node Acc {:.4f} ".format(total_loss, accuracy))
    node_results = {
        'predictions': node_predictions,
        'groundtruths': node_groundtruths,
        'loss': total_loss,
        'accuracy': accuracy
    }

    summary_writer.add_scalars("Loss", {"validation": total_loss}, epoch)
    summary_writer.add_scalars("Accuracy", {"validation": accuracy}, epoch)

    return node_results, total_loss


def train(model, epoch, data_loader, criterion, optimizer, summary_writer, device):
    # Check activated modules
    model.train()

    node_loss = []
    node_instances = 0
    correct_nodes = 0

    # train node
    for graph, node_labels, _ in tqdm(data_loader):
        node_labels = torch.tensor(node_labels, dtype=torch.long).to(device)
        graph = graph.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(True):
            node_logits = model(graph)
            n_loss = criterion(node_logits, node_labels)
            n_loss.backward()
            optimizer.step()

        node_loss.append(n_loss.item())

        _, node_preds = torch.max(node_logits, dim=1)
        correct_nodes += torch.sum(node_preds == node_labels)
        node_instances += len(node_preds)

    total_loss = np.mean(np.array(node_loss))
    accuracy = float(correct_nodes) / node_instances
    print("Epoch {:05d} | Node Loss {:.4f} | Node Acc {:.4f} |"
          .format(epoch, total_loss, accuracy))

    summary_writer.add_scalars("Loss", {"training": total_loss}, epoch)
    summary_writer.add_scalars("Accuracy", {"training": accuracy}, epoch)


def main(args):
    # Read configuration
    config = ConfigReader(args.json_config)

    if args.store_path is None:
        args.store_path = './store'

    if not os.path.isdir(args.store_path):
        os.mkdir(args.store_path)

    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    # Training data loader
    train_dataset = GraphDataset(data_path=args.train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.arrange_batch)

    # validation data loader
    valid_dataset = GraphDataset(data_path=args.valid_dataset)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  collate_fn=valid_dataset.arrange_batch)

    model = NodeClassifier(config.model)
    # model.set_device(device)

    n_classes = model.n_classes
    class_names = model.class_names

    # Check and load pretrained model
    if args.trained is not None and args.trained != 'None':
        model.load_state_dict(torch.load(args.trained, map_location=device))
        print("Restored from {}".format(args.trained))
    else:
        print('No trained model specifying ...')

    criterion = nn.CrossEntropyLoss()

    if config.optim == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optim == 'amsgrad':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, amsgrad=True)
    elif config.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = '{}/logs/{}'.format(args.store_path, current_time)
    summary_writer = SummaryWriter(log_dir)

    # Initialize
    best_node_acc = 0.
    best_node_f1 = 0.
    best_mino_class_f1 = 0.
    best_main_class_f1 = 0.
    best_val_loss = 10000.
    early_stopping_count = 0

    for epoch in range(config.nepoch):
        print('{:-^50s}'.format(''))

        # Training phase
        train(model,
              epoch,
              train_dataloader,
              criterion,
              optimizer,
              summary_writer,
              device=device)

        # Validation phase
        node_results, total_loss = validate(model,
                                            epoch,
                                            valid_dataloader,
                                            criterion,
                                            summary_writer,
                                            device=device)

        save_best = False
        node_predictions = node_results['predictions']
        node_groundtruths = node_results['groundtruths']
        node_acc = node_results['accuracy']
        node_f1_score, main_class_f1_score = calculate_confusion_matrix(node_predictions,
                                                                        node_groundtruths,
                                                                        n_classes, class_names,
                                                                        args.show_cfmatrix)
        # Save best model
        if node_acc > best_node_acc:
            best_node_acc = node_acc
            save_best = True
        if node_f1_score > best_node_f1:
            best_node_f1 = node_f1_score
            save_best = True
        # save model base on main class
        if best_main_class_f1 < main_class_f1_score:
            best_main_class_f1 = main_class_f1_score
            save_best = True
        node_prefix = '_nacc{:.04f}_nf1{:0.4f}_main_f1{:0.4f}'.format(node_acc, node_f1_score, main_class_f1_score)

        if save_best:
            print('Saving best model ...')
            torch.save(model.state_dict(),
                       os.path.join(args.store_path, 'model_epoch{:04d}_loss{:.04f}{}.pth'
                                    .format(epoch, total_loss, node_prefix)))

    summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification on graph-based documents.')
    parser.add_argument("--train_dataset", required=True, help="path to training data set")
    parser.add_argument("--valid_dataset", required=True, help="path to validation data set")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--trained", default=None, help="path to pre-trained model")
    parser.add_argument("--store_path", default=None, help="path to save trained model")
    parser.add_argument("--json_config", required=True, help="path to json file of configuration")
    parser.add_argument("--show_cfmatrix", action='store_true', help="to show confusion matrix in evaluation")
    args = parser.parse_args()
    print(args)
    main(args)

# python graph_train.py --train_dataset D:/mnt/data_source/sroie/2023-1505/train --valid_dataset D:/mnt/data_source/sroie/2023-1505/test --json_config C:/Users/withcake/Desktop/KIE_module-master/KIE_module-master/config_graph/config_node.json