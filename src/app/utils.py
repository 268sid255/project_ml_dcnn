from typing import List
from ultralytics import YOLO
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import streamlit as st
from src.yolact.yolact import Yolact


class DefectYolact:
    def get_fraction(self) -> NDArray:
        raise NotImplementedError()

    def plot_on(self, img: NDArray, color=(0, 255, 0), thickness=2) -> NDArray:
        raise NotImplementedError()


class BasePredictor:
    def predict(self, img: NDArray) -> List[DefectYolact]:
        raise NotImplementedError()


class InstanceSegmentationDefect(DefectYolact):
    def __init__(self, contour):
        self.contour = contour

    @staticmethod
    def diag(box):
        return np.linalg.norm(box[0]-box[2])

    @staticmethod
    def longest_side(box):
        return max(np.linalg.norm(box[0]-box[1]), np.linalg.norm(box[1]-box[2]))

    def get_fraction(self) -> float:
        box = cv2.minAreaRect(self.contour)
        box = cv2.boxPoints(box)
        return self.longest_side(box)

    def plot_on(self, img: NDArray, color=(0, 255, 0), thickness=2) -> NDArray:
        cv2.drawContours(image=img,
                         contours=[self.contour],
                         contourIdx=-1,
                         color=color,
                         thickness=thickness,
                         lineType=cv2.LINE_AA)
        return img


class DetectionDefect(DefectYolact):
    def __init__(self, box: List[int]):
        self.box = box

    @staticmethod
    def longest_side(box):
        return max(box[2]-box[0], box[3]-box[1])

    def get_fraction(self) -> float:
        return self.longest_side(self.box)

    def plot_on(self, img: NDArray, color=(0, 255, 0), thickness=2) -> NDArray:
        xmin, ymin, xmax, ymax = self.box
        contour = [[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax]]
        contour = np.int0(np.array(contour))
        cv2.drawContours(image=img, contours=[contour],
                         contourIdx=-1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img


def get_device(device: str):
    if device is None:
        return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return torch.device(device)


def get_contours(mask):
    contours, _ = cv2.findContours(
        mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def visualize_semantic_segmentation(img_from_camera, pred):
    return np.where(
        pred > 240,
        cv2.addWeighted(img_from_camera, 0.4, pred, 0.6, 1.0),
        img_from_camera,
    )


def plot_contours_on_img(img, defects: list):
    img_with_contours = np.copy(img)
    for defect2plot in defects:
        defect2plot.plot_on(img_with_contours)
    return img_with_contours


def get_perspective_transform(image,
                              lhs_rectangle=np.float32(
                                  [[690, 0], [1450, 0], [360, 1080], [940, 1020]]),
                              rhs_rectangle=np.float32(
                                  [[0, 0], [1920, 0], [0, 1080], [1920, 1080]]),
                              final_size=(1920, 1080)):
    # [1 2] -> [1 2]
    # [3 4] -> [3 4]
    M = cv2.getPerspectiveTransform(lhs_rectangle, rhs_rectangle)
    dst = cv2.warpPerspective(image, M, final_size)
    return dst


def load_model(model_path, model_type):
    """
    Loads a YOLO/Yolact object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.
        model_type (int): The type YOLO or Yolact
    Returns:
        A YOLO/Yolact object detection model.
    """
    if model_type == 0:
        model = YOLO(model_path)
    elif model_type == 1:
        model = Yolact()
        map_location = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model.load_weights(model_path, map_location=map_location)
        model = model.to(map_location)
    return model


def instance_detections(detected_objects_summary_list, model):
    detected_objects_summary = set()
    for obj in detected_objects_summary_list:
        detected_objects_summary.add(model.names[int(obj)])
    name_summary = ", ".join(detected_objects_summary)
    st.success(f"Обнаруженные дефекты: {name_summary}")
