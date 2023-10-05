import os
import sys
import torch
from config_graph.NODE_CLASS import BERT_FEAT_SIZE

from graph_modules.graph_submodules import GNNBaseLine, Classifier, FeatureExtractor, SemanticEncoder
from graph_modules.utils import timing

# change path to ./src before reading txt files
PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)


class NodeClassifier(GNNBaseLine):

    def __init__(self, model_config):
        super(NodeClassifier, self).__init__(model_config)

        self.sentence_encoder = SemanticEncoder(BERT_FEAT_SIZE, *self.encoder_hiddenstate)

        self.node_feat_size = self.sentence_encoder.output_size

        self.feature_extractor = FeatureExtractor(self.node_feat_size,
                                                  self.filter_type,
                                                  self.extractor_hiddens,
                                                  self.activation,
                                                  self.normalization,
                                                  self.dropout)

        self.classifier = Classifier(self.feature_extractor.gnn_layers[-1]._out_feats,
                                     self.n_classes)

    # @timing
    def __call__(self, graph):
        padded_sequence = graph.ndata["textline_features"]
        input_lens = graph.ndata["textline_lengths"]
        n_sequence = len(input_lens)
        # Packing
        packed_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence,
                                                                         input_lens.to("cpu"),
                                                                         enforce_sorted=False,
                                                                         batch_first=True)
        outputs, (h_n, c_n) = self.sentence_encoder(packed_padded_sequence)
        semantic_features = h_n.permute(1, 0, 2).reshape(n_sequence, -1)

        # Node attribute
        # h = torch.cat((boolean_features, spatial_features, semantic_features), 1)
        h = semantic_features

        # Feature extraction
        h = self.feature_extractor(graph, h)

        preds = self.classifier(h)

        return preds
