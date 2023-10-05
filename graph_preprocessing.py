import os
import cv2
import shutil
import pickle
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from graph_modules.graph_constructor import GraphConstructor
from graph_modules.word_embedding import WordEmbedding
from config_graph.NODE_CLASS import NODE_LABEL_MAP

# RANDOM_SEED = 1402
# CVAT_SOURCE_PREFIX = "/mnt/cvat_share/"
# DESTINATION_PATH_PREFIX = "/mnt/data_source/prescription_KIE/"
# XML_SOURCE = "/mnt/data_source/prescription_KIE/xml_source"
# DESTINATION_FOLDER = "2021_1005"



RANDOM_SEED = 44
DATA_FOLDER = "D:/study/dataset/sroie-2019/raw/img"
DESTINATION_PATH_PREFIX = "D:/mnt/data_source/sroie"
XML_SOURCE = "D:/study/dataset/sroie-2019/interim"
DESTINATION_FOLDER = "2023-1505"

destination_path = os.path.join(DESTINATION_PATH_PREFIX, DESTINATION_FOLDER)
if os.path.exists(destination_path):
    answer = None
    while answer not in ("yes", "no"):
        answer = input("Do you want to empty this folder (yes/no):")
        answer = "yes"
        if answer == "yes":
            for filename in os.listdir(destination_path):
                file_path = os.path.join(destination_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    print("Deleting file:", file_path)
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    print("Deleting folder:", file_path)
                    shutil.rmtree(file_path)
            print("Folder {} is empty now...".format(destination_path))
        elif answer == "no":
            print("Folder {} is remain...".format(destination_path))
        else:
            print("Please enter 'yes' or 'no'!")
else:
    os.mkdir(destination_path)
    print("Creating folder:", destination_path)

# xml_list = [os.path.join(XML_SOURCE, file) for file in os.listdir(XML_SOURCE) if file.endswith(".xml")]
df_list = [os.path.join(XML_SOURCE, file) for file in os.listdir(XML_SOURCE) if file.endswith(".csv")]

# Model setup
graph_constructor = GraphConstructor()
word_embedding = WordEmbedding()
# word_embedding.set_device("cuda:0")

for df in tqdm(df_list):
    img_name = os.path.basename(df).split('.')[0]
    img_path = os.path.join(DATA_FOLDER, f'{img_name}.jpg')
    if not os.path.exists(img_path):
        print("This {} is not existed".format(img_name))
        continue
    cv2_image = cv2.imread(img_path)
        
    dataframe = pd.read_csv(df)
    dataframe.drop(['2','3', '6', '7'], axis=1, inplace=True)
    dataframe.columns = ["xmin", "ymin", "xmax", "ymax", "text", "label"]
    dataframe['label'] = dataframe['label'].str.strip()
    dataframe['label'].fillna('other', inplace=True)
    dataframe['label'].replace('invoice', 'other', inplace=True)
    
    object_map = dataframe.copy()
 
    object_map = object_map.sort_values(['ymin', 'xmin'], ascending=[True, True])
    object_map = object_map.reset_index(drop=True)
    
    bounding_box_list = np.array(list(zip(
        object_map['xmin'].tolist(),
        object_map['ymin'].tolist(),
        object_map['xmax'].tolist(),
        object_map['ymax'].tolist())))
    text_list = object_map['text'].tolist()
    # import pdb; pdb.set_trace()
    label_list = object_map['label'].tolist()
    node_labels_integer_encode = np.array([NODE_LABEL_MAP[label] for label in label_list])

    adjacency_matrix, neighbor_distance_list = graph_constructor(bounding_box_list)
    try:
        embedding_vectors = word_embedding(text_list)
    except:
        print(img_path)
        continue

    original_info = {
        'adjacency_matrix': adjacency_matrix,
        'embedding_vectors': embedding_vectors,
        'bounding_boxes': bounding_box_list,
        'original_contents': text_list,
        'labels': node_labels_integer_encode
    }
    
    with open(os.path.join(destination_path, img_name + '.pkl'), 'wb') as f:
        pickle.dump(original_info, f)
    cv2.imwrite(os.path.join(destination_path, f'{img_name}.jpg'), cv2_image)



# current_contents = [file for file in os.listdir(DATA_FOLDER) if file.endswith(("png", "jpg"))]
# train_set, temp_set = train_test_split(current_contents, test_size=0.3, random_state=RANDOM_SEED)
# validation_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=RANDOM_SEED)
# datasets = {
#     "train": train_set,
#     "validation": validation_set,
#     "test": test_set
# }

current_contents = [file for file in os.listdir(DATA_FOLDER) if file.endswith(("png", "jpg"))]
train_set, test_set = train_test_split(current_contents, test_size=0.2, random_state=RANDOM_SEED)
datasets = {
    "train": train_set,
    "test": test_set
}

for key in datasets.keys():
    source_image_path = DATA_FOLDER
    current_destination_path = os.path.join(destination_path, key)
    if not os.path.exists(current_destination_path):
        os.mkdir(current_destination_path)
    # print(datasets.get(key))
    for file in datasets.get(key):
        shutil.copy(
            os.path.join(source_image_path, file),
            os.path.join(current_destination_path, file),
        )
        shutil.copy(
            os.path.join(destination_path, file[:-3] + 'pkl'),
            os.path.join(current_destination_path, file[:-3] + 'pkl'),
        )
