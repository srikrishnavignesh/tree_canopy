
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, Mask2FormerConfig
import os
import math
import random
import numpy as np
import json
import cv2
from torch.nn.utils import clip_grad_norm_
import time
from tqdm import tqdm
import albumentations as A
import torch
import tree_commons as tc


class ImageAugmenter:

    def __init__(self):
        self.noise = np.random.randint(0, 256, (tc.CROPPED_IMAGE_HEIGHT, tc.CROPPED_IMAGE_WIDTH, 3), dtype=np.uint8)

    def _noise_indivi_trees_randomly(self, img_arr, mask, p_indivi_survival=1 / 3):
        p_indivi_kill = 1 - p_indivi_survival
        instance_ids = np.unique(mask)
        indivi_tree_ids = instance_ids[instance_ids % 2 == 1]
        indivi_kill_no = int(p_indivi_kill * len(indivi_tree_ids))
        if indivi_kill_no == 0:
            return img_arr, mask

        random.shuffle(indivi_tree_ids)
        not_survived_indivi_trees = indivi_tree_ids[:indivi_kill_no]

        not_survived_trees_mask_bool = np.isin(mask, not_survived_indivi_trees)
        not_survived_trees_mask = (not_survived_trees_mask_bool > 0).astype(np.uint8)

        survived_trees_mask = mask * (1 - not_survived_trees_mask)

        not_survived_trees_mask_3c = cv2.cvtColor(not_survived_trees_mask, cv2.COLOR_GRAY2RGB)

        blended = img_arr * (1 - not_survived_trees_mask_3c) + self.noise * not_survived_trees_mask_3c
        return blended, survived_trees_mask

    def _add_shadows(self, img_arr, mask):

        dx = np.random.choice([-40, -30, -25, 25, 30, 40]).item()
        dy = np.random.choice([-40, -30, -25, 25, 30, 40]).item()

        M = np.float32([[1, 0, dx], [0, 1, dy]]) # type: ignore

        shadow_mask = (mask > 0).astype(np.uint8)
        shadow_mask = cv2.warpAffine(shadow_mask, M, (mask.shape[1], mask.shape[0])) # type: ignore
        shadow_mask = cv2.GaussianBlur(shadow_mask, (13, 13), 70)
        mask_3c = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2RGB)

        intensity = np.random.choice([0.45, 0.30, 0.35, 0.20, 0.25, 0.20, 0.15, 0.50])
        shadowed_arr = img_arr * (1 - intensity * mask_3c)

        return np.clip(shadowed_arr, 0, 255).astype(np.uint8)

    def augment_image(self, image_arr, mask):
        h, w = tc.CROPPED_IMAGE_HEIGHT, tc.CROPPED_IMAGE_WIDTH

        transform = A.Compose([
            A.RandomCrop(height=h, width=w, p=1.0, fill_mask=0),
            A.SquareSymmetry(p=1),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.05, 0.05),
                p=0.40
            ),
            A.MaskDropout(max_objects=(2, 20), fill=0.0, fill_mask=0.0, p=0.20),
            A.Affine(scale=(0.90, 1.1), keep_ratio=True, p=0.20),
            A.CoarseDropout(num_holes_range=(1, 5), fill_mask=0, p=0.20),
            A.GaussianBlur(blur_limit=(3, 7), p=0.30),
            A.OneOf([
                A.ToGray(p=1.0),
                A.ChannelDropout(p=0.2)
            ], p=0.20)
        ])

        out = transform(image=image_arr, mask=mask)
        aug_img_arr, aug_mask = out["image"], out["mask"]

        if np.random.uniform(0, 1) <= 0.50:
            survival = np.random.choice([1/3, 1.5/3, 2/3])
            aug_img_arr, aug_mask = self._noise_indivi_trees_randomly(aug_img_arr, aug_mask, survival)

        if np.random.uniform(0, 1) <= 0.75:
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


def get_model_input(img_arr, mask, image_id):
   
    img_arr = img_arr.astype(np.uint8)
   
    indiv_tree_mask = mask * (mask % 2 == 1).astype(np.uint8)
    indiv_tree_contours = get_contours(indiv_tree_mask)

    grp_tree_mask = mask * (mask % 2 == 0).astype(np.uint8)
    grp_tree_contours = get_contours(grp_tree_mask)

    indiv_tree_class_id = tc.LABEL_TO_CLASS_ID[tc.CLASS_INDIVIDUAL_TREE]
    grp_tree_class_id = tc.LABEL_TO_CLASS_ID[tc.CLASS_GROUP_TREES]

    mask = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.int32)
    mask.fill(255)
    instance_id_to_class_id = {}
    instance_id = 1
    for contours, class_id in zip([indiv_tree_contours, grp_tree_contours], [indiv_tree_class_id, grp_tree_class_id]):
        for contour in contours:
            polygon = contour.squeeze(1)

            box = tc.get_bounding_box(polygon)
            if box[0] >= box[2] or box[1] >= box[3]:
                continue

            cv2.fillPoly(mask, [polygon], color=instance_id) # type: ignore
            instance_id_to_class_id[instance_id] = class_id
            instance_id += 1
            
    return {'image': img_arr, 'instance_id_to_class_id': instance_id_to_class_id, 'mask': np.astype(mask, dtype=np.int32), 'image_id' : image_id}



class ImageDataCache:

    def __init__(self):
        self._filename_to_img_data = self._get_image_data_by_filename()
        self._file_type_to_filenames = self._get_filenames_by_file_type(self._filename_to_img_data.keys())

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

    def sample_train_files(self, percents, no):
        sampled_files = []
        for file_type, filenames in self._file_type_to_filenames.items():
            sorted_filenames = sorted(filenames)
            sampled_files.extend(random.choices(sorted_filenames[:-2], k=int(math.ceil(percents[file_type] * no)) ))

        random.shuffle(sampled_files)
        return sampled_files

    def sample_validation_files(self, percents, no):
        sampled_files = []
        for file_type, filenames in self._file_type_to_filenames.items():
            sorted_filenames = sorted(filenames)
            sampled_files.extend(random.choices(sorted_filenames[-2:], k=int(math.ceil(percents[file_type] * no)) ))
            
        return sampled_files

    #individual tree ids are masked with odd numbers
    #group tree ids are masked with even numbers
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
            mask = np.zeros((H, W), np.int32)

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
    

class DynamicDataset(Dataset):

    def __init__(self, no, image_data_cache):
        self._no = no
        self._img_augmenter = ImageAugmenter()
        self._image_data_cache = image_data_cache

        self._percents = {tc.PREFIX_10CM_FILE_TYPE: 0.15, tc.PREFIX_20CM_FILE_TYPE: 0.15,
                          tc.PREFIX_40CM_FILE_TYPE: 0.20, tc.PREFIX_60CM_FILE_TYPE: 0.25, tc.PREFIX_80CM_FILE_TYPE: 0.25}

        self._train_datasets = []

    def __len__(self):
        return self._no

    def on_epoch_start(self):
        self._train_datasets = self._image_data_cache.sample_train_files(self._percents, self._no)

    def __getitem__(self, indx):
        filename = self._train_datasets[indx]
        img_arr, mask = self._image_data_cache.get_image_data(filename)
        aug_img_arr, aug_mask = self._img_augmenter.augment_image(img_arr, mask)

        model_inputs = get_model_input(aug_img_arr, aug_mask, indx)

        return model_inputs
    

class StaticDataset(Dataset):
    def __init__(self, no, image_data_cache):
        self._no = no

        self._image_augmenter = ImageAugmenter()

        self._percents = {
            tc.PREFIX_10CM_FILE_TYPE: 0.20, tc.PREFIX_20CM_FILE_TYPE: 0.20,
            tc.PREFIX_40CM_FILE_TYPE: 0.20, tc.PREFIX_60CM_FILE_TYPE: 0.20,
            tc.PREFIX_80CM_FILE_TYPE: 0.20
        }

        self._sampled_files = image_data_cache.sample_validation_files(self._percents, no)

        image_id = 1
        self._sampled_datasets = []
        for filename in self._sampled_files:
            img_arr, mask = image_data_cache.get_image_data(filename)
            cropped_image, cropped_mask = self._image_augmenter.random_crop_image(img_arr, mask)

            self._sampled_datasets.append((cropped_image, cropped_mask, image_id))

            image_id += 1

    def __len__(self):
        return self._no

    def __getitem__(self, indx):
        img_arr, mask, image_id = self._sampled_datasets[indx]
        return get_model_input(img_arr, mask, image_id)
    

def collate(datasets):
    
    img_arrs = []
    masks = []
    arr_instance_id_to_class_id = []
    image_ids = []
    for dataset in datasets:
        img_arrs.append(dataset['image'])
        masks.append(dataset['mask'])
        arr_instance_id_to_class_id.append(dataset['instance_id_to_class_id'])
        image_ids.append(dataset['image_id'])
        
    return img_arrs, masks, arr_instance_id_to_class_id, image_ids


def get_pretrained_model(device):
    ckpt_dir = 'facebook/mask2former-swin-tiny-coco-instance'
    cfg = Mask2FormerConfig.from_pretrained(ckpt_dir, num_labels=2, id2label=tc.CLASS_ID_TO_LABEL, label2id=tc.LABEL_TO_CLASS_ID)
    
    cfg.backbone_config.drop_path_rate = 0.02 # type: ignore
    cfg.num_queries = 350
    cfg.backbone_config.attention_probs_dropout_prob = 0.025 # type: ignore
    cfg.backbone_config.hidden_dropout_prob = 0.025 # type: ignore
    cfg.backbone_config.dropout = 0.025 # type: ignore
    cfg.backbone_config.id2label = tc.CLASS_ID_TO_LABEL # type: ignore
    cfg.backbone_config.label2id = tc.LABEL_TO_CLASS_ID # type: ignore
    cfg.image_size = tc.CROPPED_IMAGE_HEIGHT
    cfg.class_weight = 5.0
    return Mask2FormerForUniversalSegmentation.from_pretrained(ckpt_dir, config = cfg, ignore_mismatched_sizes=True).to(device) # type: ignore

def get_image_processor():
    image_processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
    image_processor.size = {'height' : tc.CROPPED_IMAGE_HEIGHT, 'width': tc.CROPPED_IMAGE_WIDTH}
    image_processor.num_labels = 3
    image_processor.do_resize = False
    image_processor.ignore_index = 255
    return image_processor

    

def checkpoint_state(epoch, model, optimizer):
    directory = tc.MASK2FORMER_CHECKPOINT_DIR
    checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}

    new_file_name = f'checkpoint-{epoch}.pt'
    old_file_name = f'checkpoint-{epoch - 1}.pt'

    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.startswith('checkpoint-') and item != old_file_name:
            os.remove(item_path)

    file_path = os.path.join(directory, new_file_name)

    torch.save(checkpoint, file_path)

def setup_state_from_checkpoint(model, optimizer):
    directory = tc.MASK2FORMER_CHECKPOINT_DIR
    latest_epoch = -1
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.startswith('checkpoint-'):
            epoch = int(item.removeprefix('checkpoint-').removesuffix('.pt'))
            latest_epoch = max(epoch, latest_epoch)

    if latest_epoch == -1:
        print('no existing checkpoint found')
        return 0

    print(f'resuming states with {latest_epoch} checkpoint')
    checkpoint_pt = torch.load(os.path.join(directory, f'checkpoint-{latest_epoch}.pt'), map_location='cuda')
    model.load_state_dict(checkpoint_pt['model_state'])
    if 'optimizer_state' in checkpoint_pt:
        optimizer.load_state_dict(checkpoint_pt['optimizer_state'])
    

    best_val = math.inf
    if os.path.isfile(tc.MASK2FORMER_TRAIN_BEST_WEIGHT_LOC):
        ckpt = torch.load(tc.MASK2FORMER_TRAIN_BEST_WEIGHT_LOC, map_location='cpu')
        best_val = ckpt['best_val']

    return latest_epoch + 1, best_val

def load_latest_model_weights(model):
    directory = tc.MASK2FORMER_CHECKPOINT_DIR
    latest_epoch = -1
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.startswith('checkpoint-'):
            latest_epoch = int(item.removeprefix('checkpoint-').removesuffix('.pt'))

    if latest_epoch == -1:
        return

    checkpoint_pt = torch.load(os.path.join(directory, f'checkpoint-{latest_epoch}.pt'), map_location='cuda')
    model.load_state_dict(checkpoint_pt["model_state"])


class ValidationDatasetMetric:

    def __init__(self, model, validation_dataset, image_processor):
        self.model = model
        self.validation_dataset = validation_dataset
        batch_size = 1
        self.validation_dataset_dataloader = DataLoader(validation_dataset, batch_size=batch_size, in_order=False, collate_fn=collate, pin_memory=True)
        self.image_processor = image_processor


    def get_prediction_loss(self):
        self.model.eval()
        self.model.to('cuda')

        step = 0
        loss_sum = 0
        with torch.no_grad():
            for (img_arrs, masks, arr_instance_id_to_class_id, _) in self.validation_dataset_dataloader:

                processed = self.image_processor(images=img_arrs, segmentation_maps=masks,
                                instance_id_to_semantic_id=arr_instance_id_to_class_id, return_tensors="pt").data

                model_inputs = {
                                  'pixel_values': processed['pixel_values'].to('cuda'),
                                  'pixel_mask': processed['pixel_mask'].to('cuda'),
                                  'class_labels': [label.to('cuda') for label in processed["class_labels"]],  
                                  'mask_labels':  [label.to('cuda') for label in processed["mask_labels"]],
                                }

                outputs = self.model(**model_inputs)
                loss = outputs.loss
                loss_sum+= loss.item()
                step+= 1

                del outputs, processed

        avg_loss = (loss_sum / step if step > 0 else float("nan"))
        return avg_loss


    def evaluate(self):
        validation_loss = self.get_prediction_loss()
        return validation_loss


from typing import cast
def train(model, train_dataset, validation_dataset, image_processor):

    batch_size = 1
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, in_order=False, 
                        pin_memory=True)
    
    validation_dataset_metric = ValidationDatasetMetric(model, validation_dataset, image_processor)

    names = []
    for indx, (name, _) in enumerate(model.named_parameters()):
        names.append(f'{name} {indx}')


    param_indices = list(range(0, len(names)))
    trainable_params = []
    for indx, param in enumerate(model.parameters()):
        param.requires_grad = False
        if indx in param_indices:
            param.requires_grad = True
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(trainable_params, lr=2.5e-5, weight_decay=1e-4)
    best_val = math.inf

    resume = False

    start_epoch = 1
    if resume:
        start_epoch, best_val = setup_state_from_checkpoint(model, optimizer)  # type: ignore

    optimizer.param_groups[0]['lr'] = 1e-5

    end_epoch = start_epoch + 50
    accumulation_steps = 1
    
    total_instances = len(dataloader)
    for epoch in range(start_epoch, end_epoch):

        model.train()
        optimizer.zero_grad()
        loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
        start = time.perf_counter()

        step = 0
        loss_sum = 0
        
        cast(DynamicDataset, dataloader.dataset).on_epoch_start()
        for (img_arrs, masks, arr_instance_id_to_class_id, _) in loop:

            processed = image_processor(images = img_arrs, segmentation_maps = masks,
                                    instance_id_to_semantic_id = arr_instance_id_to_class_id, return_tensors = "pt").data

            model_inputs = {
                            'pixel_values': processed['pixel_values'].to('cuda'),
                            'pixel_mask': processed['pixel_mask'].to('cuda'),
                            'class_labels': [label.to('cuda') for label in processed["class_labels"]],  
                            'mask_labels':  [label.to('cuda') for label in processed["mask_labels"]],
                            }
            
            output = model(**model_inputs)

            loss_sum+= output.loss.item()
            loss = output.loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == int(math.ceil(total_instances/batch_size)):
                gradient_before_clipping = clip_grad_norm_(trainable_params, max_norm=1.5, foreach=True)
                optimizer.step()
                optimizer.zero_grad()
                loop.set_postfix({
                        'gradient': gradient_before_clipping.item(),
                        'step': step,
                        'loss' : loss.item(),
                        'learning_rates': '[' + ','.join(
                            [str(param_group['lr']) for param_group in optimizer.state_dict()['param_groups']]) + ']'
                    }
                )

            step += 1

            del model_inputs, output, loss
        
        avg_loss = loss_sum / step
        validation_loss = validation_dataset_metric.evaluate()
        if validation_loss < best_val:
            torch.save({'model_state': model.state_dict(), 'best_val': validation_loss}, tc.MASK2FORMER_TRAIN_BEST_WEIGHT_LOC)

        
        end = time.perf_counter()
        print(f'{{ epoch {epoch} avg_loss: {avg_loss} took {end - start:.3f} seconds }}')
        checkpoint_state(epoch, model, optimizer)

from sahi import DetectionModel
from typing import Any
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.predict import get_sliced_prediction
class Mask2FormerSahi(DetectionModel):
    
    def __init__(
        self,
        model_path: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
        token: str | None = None,
    ):
        self._processor = processor
        self._image_shapes: list = []
        self._token = token
        
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    @property
    def processor(self):
        return self._processor

    @property
    def image_shapes(self):
        return self._image_shapes

    @property
    def num_categories(self) -> int:
        return self.model.config.num_labels

    def load_model(self):
        self.set_model(self.model, self._processor)

    def set_model(self, model: Any, processor: Any = None, **kwargs):
        self.model = model
        self.model.to(self.device)
        self.category_mapping = self.model.config.id2label

    def perform_inference(self, image: list | np.ndarray):
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError(f'{self.model is None} {self.processor is None}')

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs["pixel_values"] = inputs.pixel_values.to(self.device)
            if hasattr(inputs, "pixel_mask"):
                inputs["pixel_mask"] = inputs.pixel_mask.to(self.device)
            
            outputs = self.model(**inputs)

        if isinstance(image, list):
            self._image_shapes = [(img.shape[0], img.shape[1]) for img in image]
        else:
            self._image_shapes = [(image.shape[0], image.shape[1])]

        n_image = len(self._image_shapes)
        target_sizes = self._image_shapes
        
        post_processed_outputs = self.processor.post_process_instance_segmentation(
                                outputs,threshold=self.confidence_threshold, mask_threshold=self.mask_threshold, 
                                return_binary_maps=True, target_sizes=target_sizes)
        
        self._original_predictions = post_processed_outputs

    def get_polygonal_predictions(self, post_processed_output) -> tuple:
       
        scores=[]
        cat_ids=[]
        #returns polygons as list of lists where inner list is flattened as [x1, y1, ...,xn,yn]
        polygonal_segments=[]
      
        segments = post_processed_output['segmentation']
        segments_info = post_processed_output['segments_info']
    
        for segment, segment_info in zip(segments, segments_info):
            mask = segment.cpu().numpy().astype(np.uint8)
            score = segment_info['score']
            lbl_indx = segment_info['label_id']
            class_id = tc.INDEX_TO_CLASS_ID[lbl_indx]
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                polygonal_segments.append([contours[0].squeeze(1).flatten().tolist()])
                    
            scores.append(score)
            cat_ids.append(class_id)
                
        return scores, cat_ids, polygonal_segments

    #peroforms batch inference on the list of slices
    #shit amount [[shift_x, shift_y],...] gives how much each slice must be shifted
    #full_shape [[width, height]] gives each slices dimension
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        
        original_predictions = self._original_predictions

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        n_image = len(original_predictions)
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
            image_height, image_width  = self.image_shapes[image_ind]
            scores, cat_ids, segments = self.get_polygonal_predictions(original_predictions[image_ind])

            object_prediction_list = []

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            
            #iterate each polygonal segment
            for ind in range(len(segments)):
                category_id = cat_ids[ind]
                segment = segments[ind]
                score = scores[ind]
                
                #8 represents the numer of x,y coords of a polygon
                #we need atleast8 to have a polygon
                if len(segment[0]) >= 8:
                    object_prediction = ObjectPrediction(
                        bbox=None,
                        segmentation=segment,
                        category_id=category_id,
                        category_name=self.category_mapping[category_id],
                        shift_amount=shift_amount,
                        score=score,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


def predict(model, image_processor, run_on_a_subset_of_eval_images):
    state = torch.load(tc.MASK2FORMER_INFERENCE_BEST_WEIGHT_LOC, map_location='cuda')
    model.load_state_dict(state['model_state'])
    sahi_model = Mask2FormerSahi(model=model, processor=image_processor, 
                            mask_threshold=0.75, confidence_threshold=0.80, 
                            device='cuda', image_size=tc.CROPPED_IMAGE_HEIGHT)
    files = []
    for filename in os.listdir(tc.EVALUATION_IMAGES_PATH):
        if filename.startswith('60cm') or not run_on_a_subset_of_eval_images:
            files.append(filename)

    files = files[:10]

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
            perform_standard_pred = False
        )
        results.append(result)
    
    for filename, result in zip(files, results):
        annotations = []
        for ann in result.to_coco_predictions():
            annotation = {tc.CLASS_KEY:ann['category_name'], 'confidence_score': ann['score'], tc.SEGMENTATION_KEY:ann['segmentation'][0]}
            annotations.append(annotation)
        
        image_data = {tc.ANNOTATIONS_KEY :  annotations}
        tc.show_img_with_annotations(tc.get_evaluation_image_arr(filename), image_data)
    


if __name__ == '__main__':
    model:Mask2FormerForUniversalSegmentation = get_pretrained_model('cuda')
    image_data_cache = ImageDataCache()
    train_dataset = DynamicDataset(1900, image_data_cache)
    validation_dataset = StaticDataset(30, image_data_cache)

    model.state_dict()['criterion.empty_weight'][1] = 5

    image_processor = get_image_processor()

    train(model, train_dataset,validation_dataset, image_processor)

    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.max_open_warning'] = 200

    # predict(model, image_processor, run_on_a_subset_of_eval_images= True)