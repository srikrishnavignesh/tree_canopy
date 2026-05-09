import mask2former
import tree_commons as tc
import torch
from sahi.predict import get_sliced_prediction

device = "cuda" if torch.cuda.is_available() else "cpu"

model = mask2former.get_pretrained_model(device)

image_processor = mask2former.get_image_processor()
state = torch.load(tc.MASK2FORMER_INFERENCE_BEST_WEIGHT_LOC, map_location=device)

#pytorch models are thread safe so it is safe to share
model.load_state_dict(state['model_state'])
model.eval()

def predict(img_arr):

    sahi_model = mask2former.Mask2FormerSahi(model=model, processor=image_processor, 
                                mask_threshold=0.50, confidence_threshold=0.75, 
                                device=device, image_size=tc.CROPPED_IMAGE_HEIGHT)
    
        
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

