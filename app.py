from flask import Flask, request, Response
import pandas as pd
import os
import cv2
import numpy as np
import yolo_app_predict
import mask2former_app_predict
import argparse

app = Flask(__name__)


@app.route('/tree_canopy/health')
def health_check():
    return 'dont worry i m there'

def get_np_array(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_arr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR_RGB)
    return img_arr


@app.route('/tree_canopy/mask2former_prediction', methods=['POST']) 
def get_mask2former_prediction():
    img_file = request.files['image'] 
    img_arr = get_np_array(img_file)

    res_img_arr = mask2former_app_predict.predict(img_arr)
    
    _, buffer = cv2.imencode(".png", cv2.cvtColor(res_img_arr, cv2.COLOR_RGB2BGR))
    return Response(buffer.tobytes(), mimetype="image/png")
    
    
@app.route('/tree_canopy/yolo_prediction', methods=['POST']) 
def get_yolo_prediction():
    img_file = request.files['image'] 
    img_arr = get_np_array(img_file)
    
    res_img_arr = yolo_app_predict.predict(img_arr)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(res_img_arr, cv2.COLOR_RGB2BGR))
    return Response(buffer.tobytes(), mimetype="image/png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the FastAPI application.")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the app on")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)