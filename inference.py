import os
import cv2
import time
import pickle
import datetime
import argparse

import dgl
import torch
import pytesseract
import numpy as np
import networkx as nx
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image

from graph_modules.word_embedding import WordEmbedding
from graph_modules.node_classifier import NodeClassifier
from graph_modules.graph_constructor import GraphConstructor
from config.NODE_CLASS import BERT_FEAT_SIZE
from graph_modules.utils import ConfigReader


def extract_textlines(tesseract_df):
    tesseract_df = tesseract_df[tesseract_df['conf'] != -1]
    line_boxes = []
    line_texts = []
    block_num = 1
    par_num = 1
    line_num = 1
    bounding_boxes = []
    text_list = []
    for idx, row in tesseract_df.iterrows():
        if (row['block_num'] != block_num) or (row["par_num"] != par_num) or (row["line_num"] != line_num):
            block_num = row['block_num']
            par_num = row['par_num']
            line_num = row['line_num']
            y_tl = min(map(lambda x: x[0], line_boxes))  # y top left
            x_tl = min(map(lambda x: x[1], line_boxes))  # x top left
            y_bt = max(map(lambda x: x[2], line_boxes))  # y bottom right
            x_bt = max(map(lambda x: x[3], line_boxes))  # x bottom right
            bounding_boxes.append([x_tl, y_tl, x_bt, y_bt])
            text_list.append(" ".join(line_texts).replace("<", "").replace("&", ""))
            line_texts = []
            line_boxes = []
        line_texts.append('{}'.format(row['text']))
        line_boxes.append((row['top'],
                           row['left'],
                           row['top'] + row['height'],
                           row['left'] + row['width']))

    return bounding_boxes, text_list


def main(args):
    # Read configuration
    config = ConfigReader(args.json_config)

    if args.store_path is None:
        args.store_path = './store'

    if not os.path.isdir(args.store_path):
        os.mkdir(args.store_path)

    # Additional submodules
    graph_constructor = GraphConstructor()
    word_embedding = WordEmbedding()

    # Tesseract configuration
    # Adding custom options
    custom_config = r'--oem 3 --psm 11 --tessdata-dir ./config/'

    # KIE configuration
    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
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

    for sample in tqdm(args.test_samples):
        img = cv2.imread(sample)
        # Text detection and OCR
        if args.mode == "cvat_label":
            item_info = pickle.load(open(sample[:-3] + "pkl", "rb"))
            adjacency_matrix = item_info.get("adjacency_matrix")
            bounding_boxes = item_info.get("bounding_boxes")
            embedding_vectors = item_info.get("embedding_vectors")
            text_list = item_info.get("original_contents")
        else:
            results = pytesseract.image_to_data(img, lang='vie', config=custom_config, output_type='data.frame')
            bounding_boxes, text_list = extract_textlines(results)
            bounding_boxes = np.array(bounding_boxes)
            adjacency_matrix, neighbor_distance_list = graph_constructor(bounding_boxes)
            embedding_vectors = word_embedding(text_list)

        # Graph Conversion
        nx_adjacency_matrix = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        dgl_graph = dgl.from_networkx(nx_adjacency_matrix)

        embedding_vectors = [torch.tensor(feats).view(-1, BERT_FEAT_SIZE) for feats in embedding_vectors]
        # Get length steps of each sentence
        input_lens = np.array([feats.size()[0] for feats in embedding_vectors])
        # Padded sequence with respect to the max-length sentence in batch.
        padded_sequence = torch.nn.utils.rnn.pad_sequence(embedding_vectors, batch_first=True)
        dgl_graph.ndata['textline_features'] = padded_sequence
        dgl_graph.ndata["textline_lengths"] = torch.tensor(input_lens)

        # Graph Convolution
        dgl_graph = dgl_graph.to(device)
        node_logits = model(dgl_graph)
        node_softmaxs = torch.nn.functional.softmax(node_logits, dim=1)
        node_probs, node_preds = torch.max(node_softmaxs, dim=1)

        # Visualization
        image_basename = os.path.basename(sample)
        detected_textline_viz = img.copy()
        for box in bounding_boxes:
            tl_x, tl_y, rb_x, rb_y = box.astype('int')
            detected_textline_viz = cv2.rectangle(detected_textline_viz, (tl_x, tl_y), (rb_x, rb_y), (0, 0, 255), 2)
        # cv2.imwrite(os.path.join(args.store_path, image_basename[:-4] + '_{}_{}'.format(args.mode, "detected") + '.png'),
        #             detected_textline_viz)

        pred_dict = {class_name: [] for class_name in class_names}
        predicted_textline_viz = detected_textline_viz.copy()
        for pred, prob, text, bbox in zip(node_preds, node_probs, text_list, bounding_boxes):
            pred_name = class_names[pred]
            if pred == len(class_names) - 1:
                continue
            pred_dict.get(pred_name).append([text, prob.cpu().item()])
            tl_x, tl_y, rb_x, rb_y = bbox.astype('int')
            predicted_textline_viz = cv2.rectangle(predicted_textline_viz, (tl_x, tl_y), (rb_x, rb_y), (0, 255, 0), 2)
        # cv2.imwrite(os.path.join(args.store_path, image_basename[:-4] + '_{}_{}'.format(args.mode, "predicted") + '.png'),
        #             predicted_textline_viz)
        
        # h, w, c = img.shape
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # thickness = 2
        # color = (0, 0, 0) 
        # classname_viz = np.ones([h, w, c], np.uint8)*255
        # x_cursor, y_cursor = (10, 10)
        # # import pdb; pdb.set_trace()
        # x_gap, y_gap = cv2.getTextSize(" ", font, font_scale, thickness)[0]
        # for key in pred_dict.keys():
        #     text_w, text_h = cv2.getTextSize(key, font, font_scale, thickness)[0]
        #     classname_viz = cv2.putText(classname_viz, key, (x_cursor, y_cursor), font, 
        #                                 font_scale, color, thickness, cv2.LINE_AA)
        #     y_cursor += text_h + y_gap
        #     for item in pred_dict.get(key):
        #         put_text = "\t{} \t {:.04f}".format(*item)
        #         text_w, text_h = cv2.getTextSize(key, font, font_scale, thickness)[0]
        #         classname_viz = cv2.putText(classname_viz, put_text, (x_cursor, y_cursor), font, 
        #                                     font_scale, color, thickness, cv2.LINE_AA)
        #         y_cursor += text_h + y_gap
        
        def put_text(text_items):
            font_size = 15
            font = ImageFont.truetype("./config/arial_1.ttf", font_size)
            x_gap, y_gap = font.getsize(" ")
            ref_par = "\n".join(text_items)
            x_ref, y_ref = font.getsize_multiline(ref_par)
            # import pdb; pdb.set_trace()
            background = Image.new("RGB", (int(x_ref*1.1), int(y_ref*1.1)), 'white')
            draw = ImageDraw.Draw(background)
            x_cursor, y_cursor = (10, 10)
            draw.multiline_text((x_cursor,y_cursor), ref_par, font=font, fill=(0, 0, 0))
            
            return background
        
        text_items = []
        for key in pred_dict.keys():
            text_items.append(key)
            for item in pred_dict.get(key):
                text_items.append("   + {} - {:.04f}".format(*item))
                
        ref_h, ref_w, ref_c = img.shape        
        classname_viz = np.ones([ref_h, ref_w, ref_c], np.uint8)*255        
        text_viz = put_text(text_items)
        text_viz = np.array(text_viz)
        viz_h, viz_w, _ = text_viz.shape
        viz_ratio = viz_h/viz_w
        ref_ratio = ref_h/ref_w
        if viz_ratio < ref_ratio:
            target_w = ref_w
            target_h = int(viz_ratio*target_w)
            x, y = 0, int((ref_h-target_h)/2)
        else:
            target_h = ref_h
            target_w = int(target_h/viz_ratio)
            x, y = int((ref_w-target_w)/2), 0

        text_viz = cv2.resize(text_viz, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        classname_viz[y:y+target_h, x:x+target_w, :] = text_viz
        cv2.imwrite(os.path.join(args.store_path, image_basename[:-4] + '_{}_{}'.format(args.mode, "predicted") + '.png'),
                    np.concatenate((predicted_textline_viz, classname_viz), axis=1))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification on graph-based documents.')
    parser.add_argument("--test_samples", required=True, nargs="+", help="path to testing sample")
    parser.add_argument("--mode", default="tesseract", help="switch between tesseract or cvat_label")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--trained", default=None, help="path to pre-trained model")
    parser.add_argument("--store_path", default=None, help="path to save trained model")
    parser.add_argument("--json_config", required=True, help="path to json file of configuration")
    args = parser.parse_args()
    print(args)
    main(args)
