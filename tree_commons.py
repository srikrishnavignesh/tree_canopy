import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


MASK2FORMER_CHECKPOINT_DIR = f'mask2former_checkpoints'

MASK2FORMER_TRAIN_BEST_WEIGHT_LOC = f'mask2former_train_best_weight.pt'

MASK2FORMER_INFERENCE_BEST_WEIGHT_LOC = f'mask2former_best_weight.pt'

YOLO11_INPUT_DIR = f'yolo11_input'
YOLO11_OUTPUT_DIR = f'yolo11_output'

TRAINING_RESULTS_OUTPUT_DIR = f'results'

TRAIN_IMAGES_PATH = f'train_images'
EVALUATION_IMAGES_PATH = f'evaluation_images'
TRAIN_ANNOTATIONS_PATH = f'train_annotations_updated.json'

YOLO_BEST_WEIGHT = f'yolo_best_weight.pt'

LOG_DIR = f'logs'

CLASS_INDIVIDUAL_TREE = 'individual_tree'
CLASS_GROUP_TREES = 'group_of_trees'

SCENE_TYPE_AGRICULTURE_PLANTATION = 'agriculture_plantation'
SCENE_TYPE_RURAL_AREA = 'rural_area'
SCENE_TYPE_URBAN_AREA = 'urban_area'
SCENE_TYPE_OPEN_FIELD = 'open_field'
SCENE_TYPE_INDUSTRIAL_AREA = 'industrial_area'

FILENAME_KEY = 'file_name'
ANNOTATIONS_KEY = 'annotations'
SEGMENTATION_KEY = 'segmentation'
IMAGES_KEY = 'images'
SCENE_TYPE_KEY = 'scene_type'
CLASS_KEY = 'class'
CONFIDENCE_KEY = 'confidence'
CM_RESOLUTION_KEY = 'cm_resolution'

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

RESIZED_IMAGE_HEIGHT = 832
RESIZED_IMAGE_WIDTH = 832

CROPPED_IMAGE_HEIGHT = 480
CROPPED_IMAGE_WIDTH = 480

PREFIX_10CM_FILE_TYPE = '10cm'
PREFIX_20CM_FILE_TYPE = '20cm'
PREFIX_40CM_FILE_TYPE = '40cm'
PREFIX_60CM_FILE_TYPE = '60cm'
PREFIX_80CM_FILE_TYPE = '80cm'
FILE_TYPES_PREFIX = [PREFIX_10CM_FILE_TYPE, PREFIX_20CM_FILE_TYPE, PREFIX_40CM_FILE_TYPE, PREFIX_60CM_FILE_TYPE,
                     PREFIX_80CM_FILE_TYPE]
SCENE_TYPES = [SCENE_TYPE_AGRICULTURE_PLANTATION, SCENE_TYPE_RURAL_AREA, SCENE_TYPE_URBAN_AREA, SCENE_TYPE_OPEN_FIELD,
               SCENE_TYPE_INDUSTRIAL_AREA]

# weights = {
#     PREFIX_10CM_FILE_TYPE: {
#         SCENE_TYPE_AGRICULTURE_PLANTATION: 2, SCENE_TYPE_RURAL_AREA: 1, SCENE_TYPE_URBAN_AREA: 1.5,
#         SCENE_TYPE_INDUSTRIAL_AREA: 1.25, SCENE_TYPE_OPEN_FIELD: 1
#     },

#     PREFIX_20CM_FILE_TYPE: {
#         SCENE_TYPE_AGRICULTURE_PLANTATION: 2.5, SCENE_TYPE_RURAL_AREA: 1.25, SCENE_TYPE_URBAN_AREA: 1.875,
#         SCENE_TYPE_INDUSTRIAL_AREA: 1.5625, SCENE_TYPE_OPEN_FIELD: 1.25
#     },
#     PREFIX_40CM_FILE_TYPE: {
#         SCENE_TYPE_AGRICULTURE_PLANTATION: 4, SCENE_TYPE_RURAL_AREA: 2, SCENE_TYPE_URBAN_AREA: 3,
#         SCENE_TYPE_INDUSTRIAL_AREA: 2.5, SCENE_TYPE_OPEN_FIELD: 2
#     },
#     PREFIX_60CM_FILE_TYPE: {
#         SCENE_TYPE_AGRICULTURE_PLANTATION: 5, SCENE_TYPE_RURAL_AREA: 2.5, SCENE_TYPE_URBAN_AREA: 3.75,
#         SCENE_TYPE_INDUSTRIAL_AREA: 3.125, SCENE_TYPE_OPEN_FIELD: 2.5
#     },

#     PREFIX_80CM_FILE_TYPE: {
#         SCENE_TYPE_AGRICULTURE_PLANTATION: 6, SCENE_TYPE_RURAL_AREA: 3, SCENE_TYPE_URBAN_AREA: 4.5,
#         SCENE_TYPE_INDUSTRIAL_AREA: 3.75, SCENE_TYPE_OPEN_FIELD: 3
#     },

# }
weights = {

    PREFIX_10CM_FILE_TYPE: 1,
    PREFIX_20CM_FILE_TYPE: 1,
    PREFIX_40CM_FILE_TYPE: 1.5,
    PREFIX_60CM_FILE_TYPE: 1.75,
    PREFIX_80CM_FILE_TYPE: 2
}

LABEL_TO_CLASS_ID = {CLASS_INDIVIDUAL_TREE: 0, CLASS_GROUP_TREES: 1}

COLOR_MAP = {1: (0, 0, 200), 2: (255, 0, 0)}

MASK_THRESHOLD = 0.5

CONTOUR_AREA = 5

CLASS_ID_TO_LABEL = {}
for k, v in LABEL_TO_CLASS_ID.items():
    CLASS_ID_TO_LABEL[v] = k

INDEX_TO_CLASS_ID = {}
for indx, (k, v) in enumerate(LABEL_TO_CLASS_ID.items()):
    INDEX_TO_CLASS_ID[indx] = v

def get_yolo11_train_images_dir():
    return f'{YOLO11_INPUT_DIR}/images/train'

def get_yolo11_train_labels_dir():
    return f'{YOLO11_INPUT_DIR}/labels/train'

def get_yolo11_val_images_dir():
    return f'{YOLO11_INPUT_DIR}/images/val'

def get_yolo11_val_labels_dir():
    return f'{YOLO11_INPUT_DIR}/labels/val'

    
def display_image(img_arr):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr)
    ax.axis("off")


def display_grey_scale_image(img_arr):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr, cmap='gray')
    ax.axis("off")


def get_train_images_file_path(filename):
    return f'{TRAIN_IMAGES_PATH}/{filename}'


def get_image_name_from_file_name(filename):
    return filename[:filename.rfind('.')]


def show_img_with_annotations_from_img_data(image_data):
    annotations = image_data['annotations']
    filename = image_data['file_name']

    image_path = f'{TRAIN_IMAGES_PATH}/{filename}'

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    colors = {'individual_tree': 'red', 'group_of_trees': 'blue'}

    for ann in annotations:
        col = colors[ann['class']]
        seg = ann['segmentation']
        pts = np.array(seg).reshape(-1, 2)
        patch = Polygon(pts, closed=True, facecolor=col, alpha=0.25, linewidth=2)
        ax.add_patch(patch)


def show_img_with_annotations(img_arr, image_data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr)
    ax.axis("off")
    colors = {'individual_tree': 'red', 'group_of_trees': 'blue'}

    for ann in image_data['annotations']:
        col = colors[ann['class']]
        seg = ann['segmentation']
        pts = np.array(seg).reshape(-1, 2)
        patch = Polygon(pts, closed=True, facecolor=col, alpha=0.25, linewidth=2)
        ax.add_patch(patch)
    plt.show()


def get_train_image_arr(filename):
    img_path = f'{TRAIN_IMAGES_PATH}/{filename}'
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def get_evaluation_image_arr(filename):
    img_path = f'{EVALUATION_IMAGES_PATH}/{filename}'
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_filename_from_image_name(image_name):
    return f'{image_name}.tif'


def get_polygon(tree_seg):
    return np.array(tree_seg, dtype=np.int32).reshape(-1, 2)


def get_bounding_box(polygon):
    x_min = float(polygon[:, 0].min())
    y_min = float(polygon[:, 1].min())
    x_max = float(polygon[:, 0].max())
    y_max = float(polygon[:, 1].max())

    return [x_min, y_min, x_max, y_max]


def overlay_image_with_instance_mask(image_arr, mask):
    ids = np.unique(mask)
    grp_trees_ids = ids[(ids > 0) & (ids % 2 == 0)]
    indivi_trees_ids = ids[ids % 2 == 1]
    
    overlay_image = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    if len(grp_trees_ids):
        grp_trees_mask = np.isin(mask, grp_trees_ids)
        grp_trees_mask_uint8 = (grp_trees_mask > 0).astype(np.uint8)
        grp_trees_mask_color = np.dstack(
            [grp_trees_mask_uint8 * 255, grp_trees_mask_uint8 * 0, grp_trees_mask_uint8 * 0])
        overlay_image = cv2.addWeighted(overlay_image, 1.0, grp_trees_mask_color, 0.25, 0)
    
    if len(indivi_trees_ids):
        indivi_trees_mask = np.isin(mask, indivi_trees_ids)
        indivi_trees_mask_uint8 = (indivi_trees_mask > 0).astype(np.uint8)
        indivi_trees_mask_color = np.dstack(
            [indivi_trees_mask_uint8 * 0, indivi_trees_mask_uint8 * 0, indivi_trees_mask_uint8 * 255])
        overlay_image = cv2.addWeighted(overlay_image, 1.0, indivi_trees_mask_color, 0.25, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    ax.axis("off")

def get_overlayed_img(img_arr, image_data):
    segments_colored_arr = img_arr.copy()
    colors = {'individual_tree': (255,0,0), 'group_of_trees': (0,0,255)}

    for ann in image_data['annotations']:
        col = colors[ann['class']]
        seg = ann['segmentation']
        pts = np.array(seg).reshape(-1, 2)
        cv2.fillPoly(segments_colored_arr, [pts], color=col)

    alpha = 0.25 
    overlay_img = cv2.addWeighted(segments_colored_arr, alpha, img_arr, 1-alpha, 0)

    return overlay_img
    
        