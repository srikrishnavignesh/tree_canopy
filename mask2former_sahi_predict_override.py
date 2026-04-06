from __future__ import annotations


import time

from functools import cmp_to_key

from sahi import ObjectPrediction
from tqdm import tqdm
from sahi.predict import filter_predictions


from sahi.logger import logger

from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.prediction import  PredictionResult
from sahi.slicing import slice_image


from sahi.utils.import_utils import check_requirements

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

LOW_MODEL_CONFIDENCE = 0.1

def get_prediction(
    image_list,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: PostprocessPredictions | None = None,
    verbose: int = 0,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
) -> list[PredictionResult]:
    """Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration
        exclude_classes_by_name: Optional[List[str]]
            None: if no classes are excluded
            List[str]: set of classes to exclude using its/their class label name/s
        exclude_classes_by_id: Optional[List[int]]
            None: if no classes are excluded
            List[int]: set of classes to exclude using one or more IDs
    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = dict()

 
    # get prediction
    time_start = time.time()
    detection_model.perform_inference(image_list)
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end



    # process prediction
    time_start = time.time()
    # works only with 1 batch
    detection_model.convert_original_predictions(
        shift_amount=shift_amount,
        full_shape=full_shape,

    )
    object_prediction_list_per_image = detection_model.object_prediction_list_per_image

  
    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    preds = []
    for image, object_prediction_list in zip(image_list, object_prediction_list_per_image):

        res = PredictionResult(
            image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
        )
        preds.append(res)

    return preds


def get_sliced_prediction(
    image,
    detection_model=None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    num_batch = 1,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int | None = None,
    auto_slice_resolution: bool = True,
    slice_export_prefix: str | None = None,
    slice_dir: str | None = None,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
) -> PredictionResult:
    
    # for profiling
    durations_in_seconds = dict()

   
    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        output_file_name=slice_export_prefix,
        output_dir=slice_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
    )
    from sahi.models.ultralytics import UltralyticsDetectionModel

    num_slices = len(slice_image_result)
    
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    if isinstance(detection_model, UltralyticsDetectionModel) and detection_model.is_obb:
        # Only NMS is supported for OBB model outputs
        postprocess_type = "NMS"

    # init match postprocess instance
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} "
            f"but given as {postprocess_type}"
        )
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    postprocess_time = 0
    time_start = time.time()

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        tqdm.write(f"Performing prediction on {num_slices} slices.")
    object_prediction_list = []
    # perform sliced prediction
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        prediction_results = get_prediction(
            image_list=image_list,
            detection_model=detection_model,
            shift_amount=shift_amount_list,
            full_shape=[[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ]]*num_batch,
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )
        for prediction_result in prediction_results:
            # convert sliced predictions to full predictions
            for object_prediction in prediction_result.object_prediction_list:
                if object_prediction:  # if not empty
                    object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # merge matching predictions during sliced prediction
        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            postprocess_time_start = time.time()
            object_prediction_list = postprocess(object_prediction_list)
            postprocess_time += time.time() - postprocess_time_start

    

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end - postprocess_time
    durations_in_seconds["postprocess"] = postprocess_time

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
        print(
            "Postprocessing performed in",
            durations_in_seconds["postprocess"],
            "seconds.",
        )

    return PredictionResult(
        image=image, object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )



