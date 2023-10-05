
import os
import sys
import torch
import torch.nn as nn
from config_graph.NODE_CLASS import N_NODE_CLASSES, NODE_CLASS_NAMES
from dgl.nn.pytorch import ChebConv, GATConv, GraphConv, SAGEConv
from graph_modules.utils import timing

# change path to ./src before reading txt files
PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)

ACTIVATION_DICTS = {
    'relu': nn.ReLU(),
    'leakyrelu': nn.LeakyReLU()
}


class GNNBaseLine(nn.Module):

    def __init__(self, model_config):
        super(GNNBaseLine, self).__init__()

        # Class info
        self.n_classes = N_NODE_CLASSES
        self.class_names = NODE_CLASS_NAMES

        # Layer configs
        self.filter_type = model_config['filter_type']
        self.extractor_hiddens = model_config['feature_extractor']
        self.encoder_hiddenstate = model_config['sentence_encoder']

        # Other config
        self.activation = model_config['activation']
        self.normalization = model_config['normalization']
        self.dropout = model_config['dropout']


class SemanticEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional):
        super(SemanticEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.encoder, self.output_size = self.build_sentence_encoder()
    
    # @timing
    def __call__(self, padded_sequences):
        # Forward
        semantic_features = self.encoder(padded_sequences)

        return semantic_features

    def build_sentence_encoder(self):
        # Create LSTM encoder layer w.r.t LSTM direction.
        encoder = nn.LSTM(self.input_size, self.hidden_size, bidirectional=self.bidirectional)
        output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        return encoder, output_size


class FeatureExtractor(nn.Module):
    def __init__(self, node_feat_size, filter_type, hidden_layers, activation, normalization, dropout):
        super(FeatureExtractor, self).__init__()
        self.node_feat_size = node_feat_size
        self.filter_type = filter_type
        self.n_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.norm_type = normalization
        self.gnn_layers, self.normalization = self.build_gnn_layers()
        self.activation = self.select_activation_function(activation)
        self.dropout = None

    # @timing
    def __call__(self, graph, node_features):
        h = node_features
        for i, hidden_layer in enumerate(self.gnn_layers):
            h = hidden_layer(graph, h)
            if self.filter_type == "gat":
                if i < self.n_layers - 1:
                    h = torch.reshape(h, (h.shape[0], -1))
                else:
                    h = torch.mean(h, dim=1)

            if self.norm_type == "instance":
                h = self.normalization[i](torch.unsqueeze(h, dim=0))
                h = torch.squeeze(h)
            if (self.norm_type == "batch") and (h.shape[0] > 1):
                h = self.normalization[i](h)

            if self.activation is not None:
                h = self.activation(h)

        return h

    def build_gnn_layers(self):
        extractor = []
        normalization = []
        in_feat_size = self.node_feat_size

        # Add GNN layer
        gnn_layer, layer_output = self.create_gnn_layer(self.filter_type,
                                                        in_feat_size,
                                                        self.hidden_layers[0])
        extractor.append(gnn_layer)
        # Add normalization layer
        normalization.append(self.create_norm_layer(self.norm_type, layer_output))

        for i in range(1, self.n_layers):
            gnn_layer, layer_output = self.create_gnn_layer(self.filter_type,
                                                            layer_output,
                                                            self.hidden_layers[i])
            extractor.append(gnn_layer)
            normalization.append(self.create_norm_layer(self.norm_type,
                                                        layer_output if i < self.n_layers - 1
                                                        else gnn_layer._out_feats))

        return nn.ModuleList(extractor), nn.ModuleList(normalization)

    @staticmethod
    def create_gnn_layer(filter_type, in_feat_size, layer_config):
        if filter_type == 'chebnet':
            # Chebyshev filter
            # Number of features and K-localized, respectively.
            out_feats_size, k_localized = layer_config
            gnn_layer = ChebConv(in_feat_size,
                                 out_feats_size,
                                 k=k_localized,
                                 bias=True, activation=None)
            layer_output = out_feats_size

        elif filter_type == 'gcn':
            # First order Chebyshev filter with constraints
            out_feats_size = layer_config
            gnn_layer = GraphConv(in_feat_size,
                                  out_feats_size,
                                  allow_zero_in_degree=True)
            layer_output = out_feats_size

        elif filter_type == 'sage':
            # GraphSAGE filter
            out_feats_size, aggregator_type = layer_config
            gnn_layer = SAGEConv(in_feat_size,
                                 out_feats_size,
                                 aggregator_type=aggregator_type)
            layer_output = out_feats_size

        elif filter_type == 'gat':
            # GAT filter
            out_feats_size, num_heads = layer_config
            gnn_layer = GATConv(in_feat_size,
                                out_feats_size,
                                num_heads,
                                residual=True,
                                allow_zero_in_degree=True)
            layer_output = out_feats_size * num_heads

        return gnn_layer, layer_output

    @staticmethod
    def select_activation_function(activation_type):
        if activation_type is None:
            return activation_type

        return ACTIVATION_DICTS.get(activation_type)

    @staticmethod
    def create_norm_layer(norm_type, feat_size):
        norm_layer = None
        # Instance normalization
        if norm_type == 'instance':
            # Axis 1 for [N, C] and 2 for [B, N, C] or -1 for the last dimension
            norm_layer = nn.InstanceNorm1d(feat_size)
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm1d(feat_size)

        return norm_layer


class Classifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.classifier = self.build_classifier()

    # @timing
    def __call__(self, h):
        preds = self.classifier(h)

        return preds

    def build_classifier(self):
        classifier = nn.Linear(self.input_size, self.n_classes)

        return classifier
