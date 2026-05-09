import tree_commons as tc
import sahi
import os
from sahi.predict import get_sliced_prediction
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model = YOLO(tc.YOLO_BEST_WEIGHT)

model.eval()
model.to(device)

def predict(img_arr):

    sahi_model = sahi.AutoDetectionModel.from_pretrained(model_type='ultralytics', model= model,
                                            confidence_threshold=0.35, device=device, mask_threshold=0.90, 
                                            image_size = tc.CROPPED_IMAGE_HEIGHT)


    result = get_sliced_prediction(
        image=img_arr,
        detection_model=sahi_model,
        slice_height=tc.CROPPED_IMAGE_HEIGHT,        
        slice_width=tc.CROPPED_IMAGE_WIDTH,         
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2, 
        batch_size=9
    )
        

   
    annotations = []
    for ann in result.to_coco_predictions():
        annotation = {
                        tc.CLASS_KEY:ann['category_name'], 'confidence_score': ann['score'], 
                        tc.SEGMENTATION_KEY:ann['segmentation'][0]
                     }
        annotations.append(annotation)
    
    image_data = {tc.ANNOTATIONS_KEY :  annotations}
    
    return tc.get_overlayed_img(img_arr, image_data)

