import math

import networkx as nx
import numpy as np
import pandas as pd


def construct_vert_line(
    src_range_x,
    src_center_y,
    dest_range_x,
    dest_center_y,
    dest_idx,
):
    is_beneath_or_above = False
    attributes = ()
    # check if horizontal range of dest lies within range
    # of source

    # case 1
    if dest_range_x[0] <= src_range_x[0] \
            and dest_range_x[1] >= src_range_x[1]:

        x_common = (src_range_x[0] + src_range_x[1]) / 2

        line_src = (x_common, src_center_y)
        line_dest = (x_common, dest_center_y)

        attributes = (dest_idx, line_src, line_dest)
        is_beneath_or_above = True

    # case 2
    elif dest_range_x[0] >= src_range_x[0] \
            and dest_range_x[1] <= src_range_x[1]:

        x_common = (dest_range_x[0] + dest_range_x[1]) / 2

        line_src = (x_common, src_center_y)
        line_dest = (x_common, dest_center_y)

        attributes = (dest_idx, line_src, line_dest)
        is_beneath_or_above = True

    # case 3
    elif dest_range_x[0] <= src_range_x[0] <= dest_range_x[1] < src_range_x[1]:

        x_common = (src_range_x[0] + dest_range_x[1]) / 2

        line_src = (x_common, src_center_y)
        line_dest = (x_common, dest_center_y)

        attributes = (dest_idx, line_src, line_dest)
        is_beneath_or_above = True

    # case 4
    elif dest_range_x[1] >= src_range_x[1] >= dest_range_x[0] > src_range_x[0]:

        x_common = (dest_range_x[0] + src_range_x[1]) / 2

        line_src = (x_common, src_center_y)
        line_dest = (x_common, dest_center_y)

        attributes = (dest_idx, line_src, line_dest)
        is_beneath_or_above = True

    return attributes, is_beneath_or_above


def construct_hori_line(
    src_range_y,
    src_center_x,
    dest_range_y,
    dest_center_x,
    dest_idx
):
    # check if vertical range of dest lies within range
    # of source
    attributes = ()

    # case 1
    if dest_range_y[0] >= src_range_y[0] \
            and dest_range_y[1] <= src_range_y[1]:

        y_common = (dest_range_y[0] + dest_range_y[1]) / 2

        line_src = (src_center_x, y_common)
        line_dest = (dest_center_x, y_common)

        attributes = (dest_idx, line_src, line_dest)

    # case 2
    if dest_range_y[0] <= src_range_y[0] < dest_range_y[1] <= src_range_y[1]:

        y_common = (src_range_y[0] + dest_range_y[1]) / 2

        line_src = (src_center_x, y_common)
        line_dest = (dest_center_x, y_common)

        attributes = (dest_idx, line_src, line_dest)

    # case 3
    if src_range_y[0] <= dest_range_y[0] < src_range_y[1] <= dest_range_y[1]:

        y_common = (dest_range_y[0] + src_range_y[1]) / 2

        line_src = (src_center_x, y_common)
        line_dest = (dest_center_x, y_common)

        attributes = (dest_idx, line_src, line_dest)

    # case 4
    if dest_range_y[0] <= src_range_y[0] \
            and dest_range_y[1] >= src_range_y[1]:

        y_common = (src_range_y[0] + src_range_y[1]) / 2

        line_src = (src_center_x, y_common)
        line_dest = (dest_center_x, y_common)

        attributes = (dest_idx, line_src, line_dest)

    return attributes


def disable_angle_mask(angle_mask_line_of_sight_, from_, to_):
    """
    mask the range of angle LoS from (from_) to (to_) is sighted
    """
    for tmpp in range(from_, to_, 1):
        angle_mask_line_of_sight_[tmpp] = False
    return angle_mask_line_of_sight_


def construct_LOS_line(src_max_pt, dst_bb, dst_idx):
    """
    src_max_pt: (xmax, ymax) point of src
    dst_bb: (xmin, ymin, xmax, ymax) of dst
    """
    attributes = ()

    dst_center_pt = ((dst_bb[0] + dst_bb[2]) / 2, (dst_bb[1] + dst_bb[3]) / 2)

    # create distance list from source_point_LoS and sort
    dst_max_pt = (dst_bb[2], dst_bb[3])
    if dst_max_pt[0] > src_max_pt[0] and dst_max_pt[1] > src_max_pt[1]:
        dist, angle = dist_n_angle_point2box(src_max_pt, dst_bb)
        # if dist > LOS_minlength_link or dist < LOS_maxlength_link:
        attributes = (dst_idx, src_max_pt, dst_center_pt, dist, angle)

    return attributes


def filter_LOS(dest_attr_LOS_sorted, df, angle_mask_line_of_sight):
    pos_priority = 0
    filtered_dest_attr_LOS = []
    for attributes in dest_attr_LOS_sorted:
        dst_idx, src_max_pt, *_ = attributes
        dst_row = df.iloc[dst_idx]
        dst_bb = [dst_row['xmin'], dst_row['ymin'], dst_row['xmax'], dst_row['ymax']]
        angle_from, angle_to = angle_line_of_sight(src_max_pt, dst_bb)

        # LOS point inside des bbox
        if angle_from == 0 and angle_to == 359:
            filtered_dest_attr_LOS.append(attributes)
            break
        # des bbox in LoS area
        elif angle_from > 0:
            if any(angle_mask_line_of_sight[angle_from: angle_to + 1]):
                filtered_dest_attr_LOS.append(attributes)
                angle_mask_line_of_sight = disable_angle_mask(angle_mask_line_of_sight, angle_from, angle_to + 1)
        # des bbox in overlap area
        elif angle_from <= 0:
            if any(angle_mask_line_of_sight[0: angle_to + 1]):
                filtered_dest_attr_LOS.insert(pos_priority, attributes)
                pos_priority += 1
                angle_mask_line_of_sight = disable_angle_mask(angle_mask_line_of_sight, 0, angle_to + 1)

    return filtered_dest_attr_LOS, angle_mask_line_of_sight


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def angle_line_of_sight(src_p, bbox):
    """
    calculate the angle LoS from src_p to bbox
    arg:
        src_p [x,y]
        bbox[xmin, ymin, xmax, ymax]
    return:
        angle_from, angle_to
        a range of angles in line of sight from angle_from, to angle_to
    """
    if is_point_inside_box(src_p, bbox):
        return 0, 359
    else:
        p1 = [bbox[0], bbox[1]]
        p2 = [bbox[0], bbox[3]]
        p3 = [bbox[2], bbox[1]]
        p4 = [bbox[2], bbox[3]]
        list_point = [p1, p2, p3, p4]
        list_angle = [angle_with_xaxis(src_p, point) for point in list_point]
        list_angle.sort()
        if 0 <= list_angle[0] <= 90 and 270 <= list_angle[3] <= 360:
            return -(360 - list_angle[2]), list_angle[1]
        else:
            return list_angle[0], list_angle[3]


def remove_duplicate_values_in_dict(graph_dict):
    ret_dict = {}
    for key in graph_dict:
        ret_dict[key] = list(set(graph_dict[key]))
    return ret_dict


def angle_with_xaxis(src_point, des_point):
    """
    return an angle value between the vector and x axis
    angle value is alway a positive value
    arg:
        src_point: [x, y]
        des_point: [a, b]
    return:
        (int)angle
    """
    vector_1 = [des_point[0] - src_point[0], des_point[1] - src_point[1]]
    vector_2 = [1, 0]  # x axis
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = math.degrees(np.arccos(dot_product))
    if vector_1[1] < 0:
        angle = 360 - angle
    return int(angle)


# TODO: simplify the logic
def dist_n_angle_point2box(src_p, bbox):
    """
    calculate distance from a point src_p to a bounding box
    arg:
        src_p: [xmin, ymin, xmax, ymax]
    return:
        distance, angle
    """
    def dist_cal(p1, p2):
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    if is_point_inside_box(src_p, bbox):
        return 0, 0
    else:
        angle_from, angle_to = angle_line_of_sight(src_p, bbox)
        if angle_from < 0:
            return abs(src_p[0] - bbox[0]), 0

        p1 = [bbox[0], bbox[1]]
        p2 = [bbox[0], bbox[3]]
        p3 = [bbox[2], bbox[1]]
        p4 = [bbox[2], bbox[3]]
        list_point = [p1, p2, p3, p4]
        list_dist = [dist_cal(src_p, p) for p in list_point]
        ids_sort = np.argsort(list_dist)
        list_dist.sort()
        angle = angle_with_xaxis(src_p, list_point[ids_sort[0]])
        return list_dist[0], angle


def is_point_inside_box(point, box):
    '''
    Is point is inside the bbox
    point: [x1, y1]
    box: [x1, y1, x2,y2]
    '''
    x, y = point
    x1, y1, x2, y2 = box
    if x1 <= x and x <= x2 and y1 <= y and y <= y2:
        return True
    else:
        return False


def find_container_index_in_dataframe(point, data_frame):
    x, y = point
    indexes = data_frame.index[
        (x >= data_frame['xmin'])
        & (x <= data_frame['xmax'])
        & (y >= data_frame['ymin'])
        & (y <= data_frame['ymax'])
    ].tolist()
    if len(indexes) > 0:
        return indexes[0]
    else:
        return -1


def correct_duplicate_link_direction_node(graph_adjacency_dict, df_merged):
    TOP = 'top'
    BOTTOM = 'bottom'
    LEFT = 'left'
    RIGHT = 'right'
    OPPOSITE_DIRECTION = {
        TOP: BOTTOM,
        BOTTOM: TOP,
        LEFT: RIGHT,
        RIGHT: LEFT
    }

    def correct_dataframe(df_merged, node_info):
        # clear the node connection
        src_id, link_direction = node_info
        df_merged.at[src_id, 'x_src_{}'.format(link_direction)] = -1
        df_merged.at[src_id, 'y_src_{}'.format(link_direction)] = -1
        df_merged.at[src_id, 'x_dest_{}'.format(link_direction)] = -1
        df_merged.at[src_id, 'y_dest_{}'.format(link_direction)] = -1
        df_merged.at[src_id, '{}_object'.format(link_direction)] = ''
        df_merged.at[src_id, '{}_length'.format(link_direction)] = 0
        df_merged.at[src_id, '{}_obj_index'.format(link_direction)] = -1

        return df_merged

    def calculate_overlap_and_lefttop_most(line_1, line_2):
        line_1_min, line_1_max = line_1
        line_2_min, line_2_max = line_2
        denumerator = min(line_1_max - line_1_min, line_2_max - line_2_min)
        numerator = min(line_1_max, line_2_max) - max(line_1_min, line_2_min)
        lefttop_most_coord = line_1_min if denumerator == line_1_max - line_1_min else line_2_min
        return numerator / denumerator, lefttop_most_coord

    def correct_adjacency_dict_and_dataframe(graph_adjacency_dict, df_merged, nodes_in_one_direction):
        # sort priority order: 1. distance (low->high)
        #                      2. overlap area(high->low)
        #                      3. left-most or top-most coordinates(low->high)
        nodes_sort_list = []
        for node_info in nodes_in_one_direction:
            node_sort_item = ()
            node_sort_item += node_info
            src_id, des_id, _, link_direction = node_info
            if link_direction == TOP or link_direction == BOTTOM:
                overlap_ratio, lefttop_most_coord = calculate_overlap_and_lefttop_most(
                    (df_merged.iloc[src_id]['xmin'], df_merged.iloc[src_id]['xmax']),
                    (df_merged.iloc[des_id]['xmin'], df_merged.iloc[des_id]['xmax']))

            elif link_direction == LEFT or link_direction == RIGHT:
                overlap_ratio, lefttop_most_coord = calculate_overlap_and_lefttop_most(
                    (df_merged.iloc[src_id]['ymin'], df_merged.iloc[src_id]['ymax']),
                    (df_merged.iloc[des_id]['ymin'], df_merged.iloc[des_id]['ymax']))
            node_sort_item += (overlap_ratio, lefttop_most_coord)
            nodes_sort_list.append(node_sort_item)

        # sorting
        # _, _, length, _, overlap_ratio, lefttop_most_coord
        nodes_sort_list = sorted(nodes_sort_list, key=lambda t: (t[2], -t[4], t[5]))

        node_to_clear_connection = nodes_sort_list[1:]  # keep the closest node by distance, clear the others
        for node_info in node_to_clear_connection:
            src_id, des_id, _, link_direction, _, _ = node_info
            graph_adjacency_dict[src_id].remove(des_id)
            df_merged = correct_dataframe(df_merged, (src_id, link_direction))
        return graph_adjacency_dict, df_merged

    # return new corrected dataframe
    G = nx.from_dict_of_lists(graph_adjacency_dict)
    adj_sparse = nx.adjacency_matrix(G)
    A = np.array(adj_sparse.todense())

    # check row to see if a row have more than one link in the same direction
    rows_to_inspect = []
    for row_id, row in enumerate(A):
        top_nodes = []  # list of (src_id, des_id, length)
        bottom_nodes = []
        left_nodes = []
        right_nodes = []
        # find column index which link to row index (value = 1)
        link_indexes = [index for index, link in enumerate(row) if link == 1]
        for col_id in link_indexes:
            # top
            if df_merged.iloc[row_id]['top_obj_index'] == col_id:
                top_nodes.append((row_id, col_id, df_merged.iloc[row_id]['top_length'], TOP))
            if df_merged.iloc[col_id]['bottom_obj_index'] == row_id:
                top_nodes.append((col_id, row_id, df_merged.iloc[col_id]['bottom_length'], BOTTOM))
            # bottom
            if df_merged.iloc[row_id]['bottom_obj_index'] == col_id:
                bottom_nodes.append((row_id, col_id, df_merged.iloc[row_id]['bottom_length'], BOTTOM))
            if df_merged.iloc[col_id]['top_obj_index'] == row_id:
                bottom_nodes.append((col_id, row_id, df_merged.iloc[col_id]['top_length'], TOP))
            # left
            if df_merged.iloc[row_id]['left_obj_index'] == col_id:
                left_nodes.append((row_id, col_id, df_merged.iloc[row_id]['left_length'], LEFT))
            if df_merged.iloc[col_id]['right_obj_index'] == row_id:
                left_nodes.append((col_id, row_id, df_merged.iloc[col_id]['right_length'], RIGHT))
            # right
            if df_merged.iloc[row_id]['right_obj_index'] == col_id:
                right_nodes.append((row_id, col_id, df_merged.iloc[row_id]['right_length'], RIGHT))
            if df_merged.iloc[col_id]['left_obj_index'] == row_id:
                right_nodes.append((col_id, row_id, df_merged.iloc[col_id]['left_length'], LEFT))

        if len(top_nodes) > 1:
            # print('top_nodes: ', top_nodes)
            graph_adjacency_dict, df_merged \
                = correct_adjacency_dict_and_dataframe(graph_adjacency_dict, df_merged, top_nodes)
        if len(bottom_nodes) > 1:
            # print('bottom_nodes: ', bottom_nodes)
            graph_adjacency_dict, df_merged \
                = correct_adjacency_dict_and_dataframe(graph_adjacency_dict, df_merged, bottom_nodes)
        if len(left_nodes) > 1:
            # print('left_nodes: ', left_nodes)
            graph_adjacency_dict, df_merged \
                = correct_adjacency_dict_and_dataframe(graph_adjacency_dict, df_merged, left_nodes)
        if len(right_nodes) > 1:
            # print('right_nodes: ', right_nodes)
            graph_adjacency_dict, df_merged \
                = correct_adjacency_dict_and_dataframe(graph_adjacency_dict, df_merged, right_nodes)

    return graph_adjacency_dict, df_merged




