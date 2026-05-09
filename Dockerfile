FROM python:3.14.3-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y build-essential python3-dev libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

COPY app.py mask2former_app_predict.py mask2former.py tree_commons.py yolo.py yolo_app_predict.py   ./
COPY mask2former_best_weight.pt yolo_best_weight.pt ./

CMD ["python3", "app.py", "--port", "7860"]
