import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml') # YOLO11
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='/root/dataset/dataset_visdrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0, 
                workers=4, 
                optimizer='SGD', # using SGD
                project='runs/train',
                name='exp',
                )