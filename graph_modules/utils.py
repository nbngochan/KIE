
import copy
import gc
import json
import os
from functools import wraps
from time import time

import cv2
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (gc.get_referrers(f)[0]["__wrapped__"], te-ts))
        return result
    return wrap


def calculate_confusion_matrix(predictions,
                               groundtruths,
                               n_classes,
                               class_names,
                               show_cfm=True):
    """
    Calculate and show confusion matrix, recall, precision and f1 score.
    Args:
        predictions: (List) Predictions from models (after arg-max function).
        groundtruths: (List) Ground-truths.
        n_classes: (int) Number of classes.
        class_names: (List) List of respective class names.
        show_cfm: (boolean) To show confusion matrix.

    Returns:
        (float) The macro average F1 score.
    """

    # Make confusion matrix with size w.r.t num of classes
    cf_matrix = confusion_matrix(groundtruths, predictions)

    existance_class_idx = list(set(groundtruths + predictions))
    existance_class_idx.sort()
    print('Total number of classes = {:^10d} | Actual number of classes = {:^10d}'.format(n_classes, len(existance_class_idx)))
    missed_classes = [class_names[i] for i in range(n_classes) if i not in existance_class_idx]
    print('Missed Classes: ', missed_classes)

    # Recall & precision
    precisions = precision_score(groundtruths, predictions, average=None, zero_division=1)
    recalls = recall_score(groundtruths, predictions, average=None, zero_division=1)
    # F1 score
    f1_scrs = f1_score(groundtruths, predictions, average=None)
    # Macro average F1 score
    avg_f1 = f1_score(groundtruths, predictions, average='weighted')

    # Show confusion matrix
    if show_cfm:
        print("Confusion Matrix")
        string_list = ['|{:*^20s}|'.format('')]
        string_list += ['{:^10s}|'.format(class_names[i][:10]) for i in range(n_classes)]
        string_list += ['|{:^10s}||{:^10s}||{:^10s}|\n'.format("Recall", "Precision", "F1 Score")]
        for i in range(n_classes):
            if i in existance_class_idx:
                actual_i = existance_class_idx.index(i)
                string_list += ['|{:^20s}|'.format(class_names[i])]
                for j in range(n_classes):
                    if j in existance_class_idx:
                        actual_j = existance_class_idx.index(j)
                        string_list += ['{:^10d}|'.format(int(cf_matrix[actual_i][actual_j]))]
                    else:
                        string_list += ['{:^10d}|'.format(0)]

                string_list += ['|{:^10f}||{:^10f}||{:^10f}|\n'.format(recalls[actual_i], precisions[actual_i], f1_scrs[actual_i])]
            else:
                string_list += ['|{:^20s}|'.format(class_names[i])]
                string_list += ['{:^10d}|'.format(0) for j in range(n_classes)]
                string_list += ['|{:^10f}||{:^10f}||{:^10f}|\n'.format(0, 0, 0)]

        print("Weighted average F1 score: {:.04f}".format(avg_f1))
        print(''.join(string_list))

    print("Confusion Matrix (short version)")
    string_list = ['|{:*^20s}|'.format('')]

    string_list += ['|{:^10s}||{:^10s}||{:^10s}||{:^15s}|\n'.format("Recall", "Precision", "F1 Score", "No. of Samples")]
    for i in range(n_classes):
        if i in existance_class_idx:
            actual_i = existance_class_idx.index(i)
            string_list += ['|{:^20s}|'.format(class_names[i])]
            num_samples = 0

            for j in range(n_classes):
                if j in existance_class_idx:
                    actual_j = existance_class_idx.index(j)
                    num_samples += int(cf_matrix[actual_i][actual_j])

            string_list += ['|{:^10f}||{:^10f}||{:^10f}||{:^15d}|\n'.format(recalls[actual_i], precisions[actual_i], f1_scrs[actual_i], num_samples)]
        else:
            string_list += ['|{:^20s}|'.format(class_names[i])]
            string_list += ['|{:^10f}||{:^10f}||{:^10f}||{:^15d}|\n'.format(0, 0, 0, 0)]
    print(''.join(string_list))
    print("Weighted average F1 score: {:.04f}".format(avg_f1))

    # Main class f1 score
    list_order_main_class = [i for i in range(len(class_names)) if class_names[i] != 'OTHER']
    main_class_f1_score = f1_score(groundtruths, predictions, labels=list_order_main_class, average='weighted')
    print("Weighted average F1 score (without OTHER): {:.04f}".format(main_class_f1_score))

    return avg_f1, main_class_f1_score


class ConfigReader:
    """
    Read json configuration.
    Args:
        json_file: (str) Path to json configuration file.
    """

    def __init__(self,
                 json_file):
        json_config = json.load(open(json_file))
        self.connect_method = json_config.get('connect_method')
        self.nepoch = json_config['nepoch']
        self.batch_size = json_config['batch_size']
        self.lr = json_config['learning_rate']
        self.weight_decay = json_config['weight_decay']
        self.model = json_config['model']
        self.optim = json_config['optim']
        self.loss_function = json_config['loss_function']


def visualize_predictions(visualization_info,
                          store,
                          show_NOLINK=False):
    """
    Draw bounding boxes and predictions onto original images.
    Args:
        visualization_info:
        store: (str) Path to directory to save visualize.
        show_NOLINK:

    Returns:

    """

    if not os.path.exists(store):
        os.mkdir(store)

    graph_name = visualization_info["graph_name"]
    graph_img = cv2.imread(graph_name)
    bounding_boxes = visualization_info["bounding_boxes"].astype('int32')
    node_results = visualization_info.get('node')
    # Node info
    if node_results is not None:
        node_class_names = node_results["class_names"]
        node_color_choices = [(0, 0, 255), (0, 255, 0), (234, 209, 116)]

        node_preds = node_results['predictions']
        node_gts = node_results['groundtruths']

        for i, (bb, pred, gt) in enumerate(zip(bounding_boxes, node_preds, node_gts)):
            if gt is not None:
                if pred != gt:
                    color_choice = (0, 0, 255)
                else:
                    color_choice = (0, 255, 0)
            else:
                color_choice = node_color_choices[pred]

            if node_class_names[pred] == node_class_names[gt] == 'OTHER':
                continue

            cv2.rectangle(graph_img, (bb[0], bb[1]), (bb[2], bb[3]), color_choice, 2)
            cv2.putText(
                graph_img,
                '{} {} {}'.format(i, node_class_names[pred],
                                    '({})'.format(node_class_names[gt])
                                    if (gt is not None) and (gt != pred) else ''),
                (bb[0], bb[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_choice, 1, cv2.LINE_AA)

    link_results = visualization_info.get('link')
    if link_results is not None:
        link_class_names = link_results["class_names"]
        link_color_choices = [(255, 255, 255), (255, 0, 0)]
        link_preds = link_results['predictions']
        link_gts = link_results['groundtruths']
        link_indices = link_results['link_indices']

        srcs, dsts = link_indices

        box_centers = [((int((bounding_boxes[src][2] + bounding_boxes[src][0])/2), int((bounding_boxes[src][3] + bounding_boxes[src][1])/2)),
                        (int((bounding_boxes[dst][2] + bounding_boxes[dst][0])/2), int((bounding_boxes[dst][3] + bounding_boxes[dst][1])/2)))
                       for src, dst in zip(srcs, dsts)]

        for i, (pred, gt, src, dst, box_center) in enumerate(zip(link_preds, link_gts, srcs, dsts, box_centers)):
            if gt is not None:
                if pred != gt:
                    color_choice = (0, 0, 255)
                else:
                    color_choice = (0, 255, 0)
            else:
                color_choice = link_color_choices[pred]

            src_point, dst_point = box_center

            # if not pred == gt == 0:  # For jointly visualizing without NOLINK class
            if gt is not None:
                if not show_NOLINK and pred == gt == 0:
                    continue
            else:
                if not show_NOLINK and pred == 0:
                    continue
            # if not show_NOLINK and pred == gt == 0:
            #     continue
            cv2.line(graph_img, src_point, dst_point, color_choice, 2)
            cv2.putText(
                graph_img,
                '{} {}'.format(link_class_names[pred],
                                '({})'.format(link_class_names[gt])
                                if (gt is not None) and (gt != pred) else ''),
                (int((src_point[0] + dst_point[0]) / 2), int((src_point[1] + dst_point[1]) / 2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color_choice, 1, cv2.LINE_AA)

        for box_center in box_centers:
            src_point, dst_point = box_center
            cv2.circle(graph_img, src_point, radius=3, color=(255, 0, 255), thickness=2)
            cv2.circle(graph_img, dst_point, radius=3, color=(255, 0, 255), thickness=2)

    cv2.imwrite(os.path.join(store, os.path.basename(graph_name)), graph_img)
