from ultralytics import YOLO
import cv2
import numpy as np
from constants import DATA_DIR, MODEL_YAML_N, MODEL_YAML_S, MODEL_YAML_M, MODEL_YAML_X, MODEL_YAML_L, MODEL_SEG_N, MODEL_SEG_S, MODEL_SEG_M, MODEL_SEG_L, MODEL_SEG_X
from argparse import ArgumentParser


def train_the_model(imgsz, epochs, batch_size, name, iou, type_pretrain_model):

    model_yaml, model_seg = ''
    if type_pretrain_model == 'n':
        model_yaml = MODEL_YAML_N
        model_seg = MODEL_SEG_N
    elif type_pretrain_model == 's':
        model_yaml = MODEL_YAML_S
        model_seg = MODEL_SEG_S
    elif type_pretrain_model == 'm':
        model_yaml = MODEL_YAML_M
        model_seg = MODEL_SEG_M
    elif type_pretrain_model == 'l':
        model_yaml = MODEL_YAML_L
        model_seg = MODEL_SEG_L
    elif type_pretrain_model == 'x':
        model_yaml = MODEL_YAML_X
        model_seg = MODEL_SEG_X

    try:
        # Load a model
        model = YOLO(model_yaml)  # build a new model from YAML
        # load a pretrained model (recommended for training)
        model = YOLO(model_seg)
        # build from YAML and transfer weights
        model = YOLO(model_yaml).load(model_seg)

        options = {
            'data': DATA_DIR,
            'imgsz': imgsz,
            'epochs': epochs,
            'batch': batch_size,
            'name': name,
            'iou': iou,
            'pretrained': True,
            'save': True,
            'overlap_mask': True
        }

        model.train(**options)  # train the model

        metrics = model.val()  # evaluate model
    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--imgsz', type=str, help='Размер изображения для обучения/Size image for train')
    parser.add_argument('--epochs', type=int,
                        help='Кол-во эпох обучения/Count epochs for train')
    parser.add_argument('--batch_size', type=int,
                        help='Размер пакета обучения/Size batch')
    parser.add_argument('--name', type=str,
                        help='Имя обученной модели/Name training model')
    parser.add_argument('--iou', type=int, help='IoU')
    parser.add_argument('--type_model_load', type=str,
                        help='Type Load Model (n,s,m,l or x)')

    args = parser.parse_args()

    imgsz = args.imgsz
    epochs = args.epochs
    batch_size = args.batch_size
    name = args.name
    iou = args.iou
    type_model_load = args.type_model_load

    train_the_model(imgsz, epochs, batch_size, name, iou, type_model_load)
