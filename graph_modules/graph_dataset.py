import copy
import os
import pickle
import sys

import dgl
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config_graph.NODE_CLASS import BERT_FEAT_SIZE

PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)


class GraphDataset(Dataset):
    def __init__(self, data_path):
        """
        Create dataset for graph data
        Args:
            data_path: (str) Path to graph data folder.
        """
        super(GraphDataset, self).__init__()
        self.data_path = data_path
        self.file_paths, self.img_paths = self.get_file_names(data_path)

        self.storage = {}

    def __len__(self):
        return len(self.file_paths)

    @torch.no_grad()
    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        image_path = self.img_paths[idx]
        item_name = os.path.basename(image_path)
        item_id = os.path.splitext(item_name)[0]

        item_info = self.storage.get(item_id)
        if self.storage.get(item_id) is None:
            item_info = pickle.load(open(image_path[:-3] + "pkl", "rb"))
            self.storage.update({
                item_id: item_info
            })

        adjacency_matrix = item_info.get("adjacency_matrix")
        nx_adjacency_matrix = nx.convert_matrix.from_numpy_array(adjacency_matrix)
        dgl_graph = dgl.from_networkx(nx_adjacency_matrix)

        node_labels = item_info.get("labels")
        bounding_boxes = item_info.get("bounding_boxes")

        embedding_vectors = item_info.get("embedding_vectors")

        return image_path, dgl_graph, node_labels, bounding_boxes, embedding_vectors

    @staticmethod
    def get_file_names(data_path):
        # import pdb; pdb.set_trace()
        file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)
                      if f.endswith(".pkl")]
        img_paths = [f.replace(".pkl", ".png") if os.path.isfile(f.replace(".pkl", ".jpg")) else None
                     for f in file_paths]
        return file_paths, img_paths

    @staticmethod
    def arrange_batch(batch):
        image_path_list = []
        bounding_boxes_list = []
        batched_graph = []
        batched_node_labels = []
        batched_embedding_vectors = []
        for item_data in batch:
            image_path, dgl_graph, node_labels, bounding_boxes, embedding_vectors = item_data

            image_path_list.append(image_path)
            bounding_boxes_list.append(bounding_boxes)

            batched_graph.append(dgl_graph)
            batched_node_labels += node_labels.tolist()
            batched_embedding_vectors += embedding_vectors

        # batch dgl graphs
        batched_graph = dgl.batch(batched_graph)
        batched_node_labels = np.array(batched_node_labels, dtype='float32')

        semantic_feats_batch = [torch.tensor(feats).view(-1, BERT_FEAT_SIZE) for feats in batched_embedding_vectors]
        # Get length steps of each sentence
        input_lens = np.array([feats.size()[0] for feats in semantic_feats_batch])
        # Padded sequence with respect to the max-length sentence in batch.
        padded_sequence = torch.nn.utils.rnn.pad_sequence(semantic_feats_batch, batch_first=True)

        batched_graph.ndata['textline_features'] = padded_sequence
        batched_graph.ndata["textline_lengths"] = torch.tensor(input_lens)

        batch_further_info = {
            'image_paths': image_path_list,
            'bboxes': bounding_boxes_list
        }

        return batched_graph, batched_node_labels, batch_further_info


if __name__ == '__main__':
    dataset = GraphDataset("D:/mnt/data_source/sroie/2023-1505/train")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=dataset.arrange_batch)
    for batch in dataloader:
        print(batch)