from ultralytics import YOLO
import tree_commons as tc
from PIL import Image
import os
import json
import math
import random
import albumentations as A
import numpy as np
import cv2
import shutil
import sahi
from sahi.predict import get_sliced_prediction

class ImageAugmenter:

    def __init__(self):
        self.noise = np.random.randint(0, 256, (tc.CROPPED_IMAGE_HEIGHT, tc.CROPPED_IMAGE_WIDTH, 3), dtype=np.uint8)

    def _noise_indivi_trees_randomly(self, img_arr, mask, p_indivi_survival=1 / 3):
        p_indivi_kill = 1 - p_indivi_survival
        instance_ids = np.unique(mask)
        indivi_kill_no = int(p_indivi_kill * len(instance_ids))
        if indivi_kill_no == 0:
            return img_arr, mask

        indivi_tree_ids = instance_ids[instance_ids % 2 == 1]

        random.shuffle(indivi_tree_ids)
        not_survived_indivi_trees = indivi_tree_ids[:indivi_kill_no]

        not_survived_trees_mask_bool = np.isin(mask, not_survived_indivi_trees)
        not_survived_trees_mask = (not_survived_trees_mask_bool > 0).astype(np.uint8)

        survived_trees_mask = mask * (1 - not_survived_trees_mask)

        not_survived_trees_mask_3c = cv2.cvtColor(not_survived_trees_mask, cv2.COLOR_GRAY2RGB)

        blended = img_arr * (1 - not_survived_trees_mask_3c) + self.noise * not_survived_trees_mask_3c
        return blended, survived_trees_mask

    def _add_shadows(self, img_arr, mask):

        dis = [-25, -20, -10, -5, 5, 10, 20, 25]
        dx = np.random.choice(dis)
        dy = np.random.choice(dis)

        M = np.float32([[1, 0, dx], [0, 1, dy]]) # type: ignore
        shadow_mask = (mask > 0).astype(np.uint8)
        shadow_mask = cv2.warpAffine(shadow_mask, M, (mask.shape[1], mask.shape[0])) # type: ignore
        shadow_mask = cv2.GaussianBlur(shadow_mask, (13, 13), 70)
        mask_3c = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2RGB)

        intensity = np.random.choice([0.20, 0.25, 0.30, 0.45,0.50])
        shadowed_arr = img_arr * (1 - intensity * mask_3c)

        return np.clip(shadowed_arr, 0, 255).astype(np.uint8)

    def augment_image(self, image_arr, mask):
        h, w = tc.CROPPED_IMAGE_HEIGHT, tc.CROPPED_IMAGE_WIDTH

        transform = A.Compose([
            A.RandomCrop(height=h, width=w, p=1.0, fill_mask=0),
            A.SquareSymmetry(p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.05, 0.05),
                p=0.10
            ),
            A.MaskDropout(max_objects=(2, 10), fill=0.0, fill_mask=0.0, p=0.15),
            A.Affine(scale=(0.90, 1.1), keep_ratio=True, p = 0.15),
            A.CoarseDropout(num_holes_range=(1, 3), fill_mask=0, p=0.10),
            A.GaussianBlur(blur_limit=(3, 5), p=0.20),
            A.OneOf([
                A.ToGray(p=1.0),
                A.ChannelDropout(p=0.2)
            ], p=0.20),
        ])

        out = transform(image=image_arr, mask=mask)
        aug_img_arr, aug_mask = out["image"], out["mask"]

        if np.random.uniform(0, 1) <= 0.50:
            survival = np.random.choice([0.5/3, 1/3, 1.5/3])
            aug_img_arr, aug_mask = self._noise_indivi_trees_randomly(aug_img_arr, aug_mask, survival)

        if np.random.uniform(0, 1) <= 0.50:
            aug_img_arr = self._add_shadows(aug_img_arr, aug_mask)

        return aug_img_arr, aug_mask

    def random_crop_image(self, image_arr, mask):
        h, w = tc.CROPPED_IMAGE_HEIGHT, tc.CROPPED_IMAGE_WIDTH

        transform = A.Compose([
            A.RandomCrop(height=h, width=w, p=1.0, fill_mask=0),
            A.SquareSymmetry(p=0.25)
        ])

        out = transform(image=image_arr, mask=mask)
        cropped_img, cropped_mask = out["image"], out["mask"]

        return cropped_img, cropped_mask

class ImageDataCache:

    def __init__(self, no_samples_for_val = 2):
        self._filename_to_img_data = self._get_image_data_by_filename()
        self._file_type_to_filenames = self._get_filenames_by_file_type(self._filename_to_img_data.keys())
        self._no_samples_for_val = no_samples_for_val

    def _get_filenames_by_file_type(self, filenames):
        file_type_to_filenames = {}
        for prefix in tc.FILE_TYPES_PREFIX:
            file_type_to_filenames[prefix] = []

        for filename in filenames:
            for prefix in tc.FILE_TYPES_PREFIX:
                if filename.startswith(prefix):
                    file_type_to_filenames[prefix].append(filename)
                    break

        return file_type_to_filenames

    def sample_files(self, no, percents, train):
        sampled_files = []
        for file_type, filenames in self._file_type_to_filenames.items():
            sorted_filenames = sorted(filenames)
            if train:
                filesubset = sorted_filenames[0:]
            else:
                filesubset = sorted_filenames[-self._no_samples_for_val:]
            
            count = int(math.ceil(percents[file_type] * no))
            sampled_files.extend(random.choices(filesubset, k=count))
        return sampled_files


    def _get_image_data_by_filename(self):
        H, W = tc.IMAGE_HEIGHT, tc.IMAGE_WIDTH

        with open(tc.TRAIN_ANNOTATIONS_PATH, 'r') as file:
            data = json.load(file)

        filename_to_img_data = {}
        images_data = data[tc.IMAGES_KEY]

        for image_data in images_data:
            filename = image_data[tc.FILENAME_KEY]
            img_arr = tc.get_train_image_arr(filename)

            indivi_tree_instance_id = 1
            grp_tree_instance_id = 2
            mask = np.zeros((H, W), np.uint16)

            annotations = image_data[tc.ANNOTATIONS_KEY]

            for annotation in annotations:
                polygon = tc.get_polygon(annotation[tc.SEGMENTATION_KEY])

                if annotation[tc.CLASS_KEY] == tc.CLASS_INDIVIDUAL_TREE:
                    instance_id = indivi_tree_instance_id
                    indivi_tree_instance_id += 2
                else:
                    instance_id = grp_tree_instance_id
                    grp_tree_instance_id += 2

                cv2.fillPoly(mask, [polygon], instance_id) # type: ignore

            filename_to_img_data[filename] = (img_arr, mask)

        return filename_to_img_data

    def get_image_data(self, filename):
        return self._filename_to_img_data[filename]

def get_contours(mask):
    all_contours = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[1:]
    for instance_id in instance_ids:
        mask_bin = (mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_contours.append(contours[0])
    return all_contours

def get_class_to_polygons_map(img_arr, mask):

    H = tc.CROPPED_IMAGE_HEIGHT
    W = tc.CROPPED_IMAGE_WIDTH
    
    indiv_tree_mask = mask * (mask % 2 == 1).astype(np.uint8)
    indiv_tree_contours = get_contours(indiv_tree_mask)

    grp_tree_mask = mask * (mask % 2 == 0).astype(np.uint8)
    grp_tree_contours = get_contours(grp_tree_mask)

    indiv_tree_class_id = tc.LABEL_TO_CLASS_ID[tc.CLASS_INDIVIDUAL_TREE]
    grp_tree_class_id = tc.LABEL_TO_CLASS_ID[tc.CLASS_GROUP_TREES]

    class_to_polygons_map = {indiv_tree_class_id :[], grp_tree_class_id:[]}
    for contours, class_id in zip([indiv_tree_contours, grp_tree_contours], [indiv_tree_class_id, grp_tree_class_id]):
        for contour in contours:
            polygon = contour.squeeze(1).astype(np.float32)
            
            polygon[:,0] /= W
            polygon[:,1] /= H
            
            class_to_polygons_map[class_id].append(polygon)

    return class_to_polygons_map

class ImageDataset():

    def __init__(self, image_data_cache, no, percents, train=True):
        self._no = no
        self._img_augmenter = ImageAugmenter()
        self._image_data_cache = image_data_cache
        self._percents = percents
        self._train = train
        self._filenames = self._image_data_cache.sample_files(self._no, self._percents, train)
        
            
    def __len__(self):
        return self._no

   
    def set_image_in_yolo_input_dir(self, indx):
        filename = self._filenames[indx]
        img_arr, mask = self._image_data_cache.get_image_data(filename)

        if self._train:
            aug_img_arr, aug_mask = self._img_augmenter.augment_image(img_arr, mask)
            image_output_dir = tc.get_yolo11_train_images_dir()
            labels_output_dir = tc.get_yolo11_train_labels_dir()
        else:
            aug_img_arr, aug_mask = self._img_augmenter.random_crop_image(img_arr, mask)
            image_output_dir = tc.get_yolo11_val_images_dir()
            labels_output_dir = tc.get_yolo11_val_labels_dir()

        class_to_polygons_map = get_class_to_polygons_map(aug_img_arr, aug_mask)

        image_id = indx
        image_name = tc.get_image_name_from_file_name(filename)
        image_output_name = f'{image_name}_{image_id}'

        
        image_output_filename = f'{image_output_dir}/{image_output_name}.png'
        os.makedirs(image_output_dir, exist_ok=True)
        img = Image.fromarray(aug_img_arr)
        img.save(image_output_filename)
        
        labels_output_filename = f'{labels_output_dir}/{image_output_name}.txt'

        txt_string = []
        for class_id, polygons in class_to_polygons_map.items():
            for polygon in polygons:
                str_polygons = ' '.join([f'{x} {y}' for x,y in polygon])
                txt_string.append(f'{class_id} {str_polygons}')

        string = '\n'.join(txt_string)
        os.makedirs(labels_output_dir, exist_ok=True)
        with open(labels_output_filename, 'w') as f:
            f.write(string)

def train():

    if os.path.isdir(tc.YOLO11_INPUT_DIR):
        shutil.rmtree(tc.YOLO11_INPUT_DIR)

    train_percents = {tc.PREFIX_10CM_FILE_TYPE: 0.15, tc.PREFIX_20CM_FILE_TYPE: 0.15,
                    tc.PREFIX_40CM_FILE_TYPE: 0.20, tc.PREFIX_60CM_FILE_TYPE: 0.25, 
                    tc.PREFIX_80CM_FILE_TYPE: 0.25}

    validation_percents = {tc.PREFIX_10CM_FILE_TYPE: 0.20, tc.PREFIX_20CM_FILE_TYPE: 0.20,
                        tc.PREFIX_40CM_FILE_TYPE: 0.20, tc.PREFIX_60CM_FILE_TYPE: 0.20, 
                        tc.PREFIX_80CM_FILE_TYPE: 0.20}

    image_data_cache = ImageDataCache()
    train_dataset = ImageDataset(image_data_cache, 2500, train_percents, train=True)
    valid_dataset = ImageDataset(image_data_cache, 180,  validation_percents, train=False)

    for indx in range(len(train_dataset)):
        train_dataset.set_image_in_yolo_input_dir(indx)
    for indx in range(len(valid_dataset)):
        valid_dataset.set_image_in_yolo_input_dir(indx)

    model = YOLO('yolo11x-seg.pt')
    res = model.train(data='data.yaml', epochs=100, batch=8, imgsz=tc.CROPPED_IMAGE_HEIGHT, save_period=10, cache=True, 
                      device=0, workers=8, name='medium_run', project=tc.YOLO11_OUTPUT_DIR, exist_ok=True, 
                      max_det=350, mask_ratio=2, classes=[0, 1], close_mosaic = 20, 
                      mosaic=0.20, hsv_s=0.10, hsv_h=0.10, translate=0.0)
    
    print(res)


def predict(run_on_a_subset_of_eval_images):
    sahi_model = sahi.AutoDetectionModel.from_pretrained(model_type='ultralytics',
                                             model_path=tc.YOLO_BEST_WEIGHT,
                                             config_path='args.yaml',
                                             confidence_threshold=0.10, device='cpu',
                                             image_size = tc.CROPPED_IMAGE_HEIGHT)
    files = []
    for filename in os.listdir(tc.EVALUATION_IMAGES_PATH):
        if filename.startswith('60cm') or not run_on_a_subset_of_eval_images:
            files.append(filename)


    results = []
    for indx, filename in enumerate(files):
        print(f'remaining is {len(files) - indx}')
        filepath = os.path.join(tc.EVALUATION_IMAGES_PATH, filename)
        result = get_sliced_prediction(
            image=filepath,
            detection_model=sahi_model,
            slice_height=tc.CROPPED_IMAGE_HEIGHT,        
            slice_width=tc.CROPPED_IMAGE_WIDTH,         
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2, 
        )
        results.append(result)
    
    for filename, result in zip(files, results):
        annotations = []
        for ann in result.to_coco_predictions():
            annotation = {tc.CLASS_KEY:ann['category_name'], 'confidence_score': ann['score'], tc.SEGMENTATION_KEY:ann['segmentation'][0]}
            annotations.append(annotation)
        
        image_data = {tc.ANNOTATIONS_KEY :  annotations}
        tc.show_img_with_annotations(tc.get_evaluation_image_arr(filename), image_data)

    
# train()

predict(True)

