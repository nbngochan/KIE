
import copy
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

# from other modules
from graph_modules import grapher_utils as utils
from graph_modules.utils import timing

# change path to ./src
PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PYTHON_PATH)
sys.path.insert(0, PYTHON_PATH)


class GraphConstructor(object):

    def __init__(self):
        super().__init__()

    @timing
    def __call__(self, data):
        graph_adjacency_dict, neighbor_distance_list = self.connect(data)

        G = nx.from_dict_of_lists(graph_adjacency_dict)
        adj_sparse = nx.adjacency_matrix(G)
        adj_matrix = np.array(adj_sparse.todense())

        return adj_matrix, neighbor_distance_list

    @staticmethod
    def connect(bounding_box_list):
        # convert bounding_box_list to dataframe
        bounding_box_dataframe = pd.DataFrame(bounding_box_list, columns=["xmin", "ymin", "xmax", "ymax"])

        df = bounding_box_dataframe.reset_index()

        # initialize empty df to store plotting coordinates
        df_plot = pd.DataFrame()

        # initialize empty lists to store coordinates and distances
        # ================== vertical======================================== #
        distances_top, nearest_dest_ids_top = [], []
        distances_bottom, nearest_dest_ids_bottom = [], []

        x_src_coords_top, y_src_coords_top, x_dest_coords_top, y_dest_coords_top = [], [], [], []
        x_src_coords_bottom, y_src_coords_bottom, x_dest_coords_bottom, y_dest_coords_bottom = [], [], [], []

        # ======================= horizontal ================================ #
        lengths_right, nearest_dest_ids_right = [], []
        lengths_left, nearest_dest_ids_left = [], []

        x_src_coords_right, y_src_coords_right, x_dest_coords_right, y_dest_coords_right = [], [], [], []
        x_src_coords_left, y_src_coords_left, x_dest_coords_left, y_dest_coords_left = [], [], [], []

        for src_idx, src_row in df.iterrows():

            # ================= vertical ======================= #
            src_range_x = (src_row['xmin'], src_row['xmax'])
            src_center_y = (src_row['ymin'] + src_row['ymax']) / 2

            dest_attr_top = []
            dest_attr_bottom = []

            # ================= horizontal ===================== #
            src_range_y = (src_row['ymin'], src_row['ymax'])
            src_center_x = (src_row['xmin'] + src_row['xmax']) / 2

            dest_attr_left = []
            dest_attr_right = []

            # ================= iterate over destination objects ===================== #
            for dest_idx, dest_row in df.iterrows():
                # flag to signal whether the destination object is below source
                is_beneath_or_above = False
                if not src_idx == dest_idx:
                    # ==================== vertical ==========================#
                    dest_range_x = (dest_row['xmin'], dest_row['xmax'])
                    dest_center_y = (dest_row['ymin'] + dest_row['ymax']) / 2

                    height = abs(dest_center_y - src_center_y)

                    # CASE Word BOTTOM
                    if dest_center_y > src_center_y:
                        attributes, is_beneath_or_above = utils.construct_vert_line(
                            src_range_x,
                            src_center_y,
                            dest_range_x,
                            dest_center_y,
                            dest_idx
                        )
                        if attributes:
                            attributes += (height, )
                            dest_attr_bottom.append(attributes)
                    # CASE Word TOP
                    elif dest_center_y < src_center_y:
                        attributes, is_beneath_or_above = utils.construct_vert_line(
                            src_range_x,
                            src_center_y,
                            dest_range_x,
                            dest_center_y,
                            dest_idx
                        )
                        if attributes:
                            attributes += (height, )
                            dest_attr_top.append(attributes)

                if not is_beneath_or_above:
                    # ======================= horizontal ==================== #
                    dest_range_y = (dest_row['ymin'], dest_row['ymax'])
                    dest_center_x = (dest_row['xmin'] + dest_row['xmax']) / 2

                    # NOTE: get length from destination center to source center (unsure about this)
                    length = abs(dest_center_x - src_center_x)
                    # if dest_center_x > src_center_x:
                    # 	length = dest_center_x - src_center_x
                    # else:
                    # 	length = 0

                    # CASE Word RIGHT
                    if dest_center_x > src_center_x:
                        attributes = utils.construct_hori_line(
                            src_range_y,
                            src_center_x,
                            dest_range_y,
                            dest_center_x,
                            dest_idx
                        )
                        if attributes:
                            attributes += (length, )
                            dest_attr_right.append(attributes)
                    # CASE Word LEFT
                    elif dest_center_x < src_center_x:
                        attributes = utils.construct_hori_line(
                            src_range_y,
                            src_center_x,
                            dest_range_y,
                            dest_center_x,
                            dest_idx,
                        )
                        if attributes:
                            attributes += (length, )
                            dest_attr_left.append(attributes)

            # sort list of destination attributes by height/length at position 3 in tuple
            dest_attr_top_sorted = sorted(dest_attr_top, key=lambda x: x[3])
            dest_attr_bottom_sorted = sorted(dest_attr_bottom, key=lambda x: x[3])
            dest_attr_left_sorted = sorted(dest_attr_left, key=lambda x: x[3])
            dest_attr_right_sorted = sorted(dest_attr_right, key=lambda x: x[3])

            # append the index and source and destination coords to draw line
            # ==================== vertical ================================= #
            if len(dest_attr_bottom_sorted) == 0:
                nearest_dest_ids_bottom.append(-1)
                x_src_coords_bottom.append(-1)
                y_src_coords_bottom.append(-1)
                x_dest_coords_bottom.append(-1)
                y_dest_coords_bottom.append(-1)
                distances_bottom.append(0)
            else:
                nearest_dest_ids_bottom.append(dest_attr_bottom_sorted[0][0])
                x_src_coords_bottom.append(dest_attr_bottom_sorted[0][1][0])
                y_src_coords_bottom.append(dest_attr_bottom_sorted[0][1][1])
                x_dest_coords_bottom.append(dest_attr_bottom_sorted[0][2][0])
                y_dest_coords_bottom.append(dest_attr_bottom_sorted[0][2][1])
                distances_bottom.append(dest_attr_bottom_sorted[0][3])
            if len(dest_attr_top_sorted) == 0:
                nearest_dest_ids_top.append(-1)
                x_src_coords_top.append(-1)
                y_src_coords_top.append(-1)
                x_dest_coords_top.append(-1)
                y_dest_coords_top.append(-1)
                distances_top.append(0)
            else:
                nearest_dest_ids_top.append(dest_attr_top_sorted[0][0])
                x_src_coords_top.append(dest_attr_top_sorted[0][1][0])
                y_src_coords_top.append(dest_attr_top_sorted[0][1][1])
                x_dest_coords_top.append(dest_attr_top_sorted[0][2][0])
                y_dest_coords_top.append(dest_attr_top_sorted[0][2][1])
                distances_top.append(dest_attr_top_sorted[0][3])

            # ========================== horizontal ========================= #
            if len(dest_attr_left_sorted) == 0:
                nearest_dest_ids_left.append(-1)
                x_src_coords_left.append(-1)
                y_src_coords_left.append(-1)
                x_dest_coords_left.append(-1)
                y_dest_coords_left.append(-1)
                lengths_left.append(0)
            else:
                # try and except for the cases where there are vertical connections
                # still to be made but all horizontal connections are accounted for
                try:
                    nearest_dest_ids_left.append(dest_attr_left_sorted[0][0])
                except:
                    nearest_dest_ids_left.append(-1)
                try:
                    x_src_coords_left.append(dest_attr_left_sorted[0][1][0])
                except:
                    x_src_coords_left.append(-1)
                try:
                    y_src_coords_left.append(dest_attr_left_sorted[0][1][1])
                except:
                    y_src_coords_left.append(-1)
                try:
                    x_dest_coords_left.append(dest_attr_left_sorted[0][2][0])
                except:
                    x_dest_coords_left.append(-1)
                try:
                    y_dest_coords_left.append(dest_attr_left_sorted[0][2][1])
                except:
                    y_dest_coords_left.append(-1)
                try:
                    lengths_left.append(dest_attr_left_sorted[0][3])
                except:
                    lengths_left.append(0)

            if len(dest_attr_right_sorted) == 0:
                nearest_dest_ids_right.append(-1)
                x_src_coords_right.append(-1)
                y_src_coords_right.append(-1)
                x_dest_coords_right.append(-1)
                y_dest_coords_right.append(-1)
                lengths_right.append(0)
            else:
                # try and except for the cases where there are vertical connections
                # still to be made but all horizontal connections are accounted for
                try:
                    nearest_dest_ids_right.append(dest_attr_right_sorted[0][0])
                except:
                    nearest_dest_ids_right.append(-1)
                try:
                    x_src_coords_right.append(dest_attr_right_sorted[0][1][0])
                except:
                    x_src_coords_right.append(-1)
                try:
                    y_src_coords_right.append(dest_attr_right_sorted[0][1][1])
                except:
                    y_src_coords_right.append(-1)
                try:
                    x_dest_coords_right.append(dest_attr_right_sorted[0][2][0])
                except:
                    x_dest_coords_right.append(-1)
                try:
                    y_dest_coords_right.append(dest_attr_right_sorted[0][2][1])
                except:
                    y_dest_coords_right.append(-1)
                try:
                    lengths_right.append(dest_attr_right_sorted[0][3])
                except:
                    lengths_right.append(0)

        # ==================== vertical ===================================== #
        # add distances column
        df['top_length'] = distances_top
        df['bottom_length'] = distances_bottom

        # add column containing index of destination object
        df['top_obj_index'] = nearest_dest_ids_top
        df['bottom_obj_index'] = nearest_dest_ids_bottom

        # add coordinates for plotting
        df_plot['x_src_top'] = x_src_coords_top
        df_plot['y_src_top'] = y_src_coords_top
        df_plot['x_dest_top'] = x_dest_coords_top
        df_plot['y_dest_top'] = y_dest_coords_top
        df_plot['x_src_bottom'] = x_src_coords_bottom
        df_plot['y_src_bottom'] = y_src_coords_bottom
        df_plot['x_dest_bottom'] = x_dest_coords_bottom
        df_plot['y_dest_bottom'] = y_dest_coords_bottom

        # ==================== horizontal =================================== #

        # add lengths_right column
        df['left_length'] = lengths_left
        df['right_length'] = lengths_right

        # add column containing index of destination object
        df['left_obj_index'] = nearest_dest_ids_left
        df['right_obj_index'] = nearest_dest_ids_right

        # add coordinates for plotting
        df_plot['x_src_left'] = x_src_coords_left
        df_plot['y_src_left'] = y_src_coords_left
        df_plot['x_dest_left'] = x_dest_coords_left
        df_plot['y_dest_left'] = y_dest_coords_left
        df_plot['x_src_right'] = x_src_coords_right
        df_plot['y_src_right'] = y_src_coords_right
        df_plot['x_dest_right'] = x_dest_coords_right
        df_plot['y_dest_right'] = y_dest_coords_right

        # ==================== concat df and df_plot =================================== #

        df_merged = pd.concat([df, df_plot], axis=1)

        # convert dataframe to dict:
        # {src_id: dest_1, dest_2, ..}
        graph_adjacency_dict = {}
        for src_id, row in df_merged.iterrows():
            graph_adjacency_dict[src_id] = []
            if row['top_obj_index'] != -1:
                graph_adjacency_dict[src_id].append(row['top_obj_index'])
            if row['bottom_obj_index'] != -1:
                graph_adjacency_dict[src_id].append(row['bottom_obj_index'])
            if row['left_obj_index'] != -1:
                graph_adjacency_dict[src_id].append(row['left_obj_index'])
            if row['right_obj_index'] != -1:
                graph_adjacency_dict[src_id].append(row['right_obj_index'])

        # find node that have more than 2 link in one direction and correct dataframe
        graph_adjacency_dict, df_merged = utils.correct_duplicate_link_direction_node(graph_adjacency_dict, df_merged)
        graph_adjacency_dict = copy.deepcopy(graph_adjacency_dict)

        neighbor_distance_list = np.zeros((len(df_merged.index), 4), dtype=float)
        # correct neighbor distance
        for src_id, row in df_merged.iterrows():
            top_obj_index = row['top_obj_index']
            if top_obj_index != -1:
                neighbor_distance_list[src_id][0] = row['top_length']
                neighbor_distance_list[top_obj_index][1] = row['top_length']
            bottom_obj_index = row['bottom_obj_index']
            if bottom_obj_index != -1:
                neighbor_distance_list[src_id][1] = row['bottom_length']
                neighbor_distance_list[bottom_obj_index][0] = row['bottom_length']
            left_obj_index = row['left_obj_index']
            if left_obj_index != -1:
                neighbor_distance_list[src_id][2] = row['left_length']
                neighbor_distance_list[left_obj_index][3] = row['left_length']
            right_obj_index = row['right_obj_index']
            if right_obj_index != -1:
                neighbor_distance_list[src_id][3] = row['right_length']
                neighbor_distance_list[right_obj_index][2] = row['right_length']
        # Since the values are increasing from left to right and from top to bottom,
        # so left_length and top_length are negative while right_length and bottom_length are positive.
        neighbor_distance_list = [
            (-top_l if top_l > 0 else 0.0, bottom_l, -left_l if left_l > 0 else 0.0, right_l)
            for (top_l, bottom_l, left_l, right_l) in neighbor_distance_list]

        return graph_adjacency_dict, neighbor_distance_list
