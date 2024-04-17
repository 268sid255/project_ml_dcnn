import PIL
import streamlit as st
from pathlib import Path
from src.app.yolo_seg import YOLODetector
from constants import WEIGHTS_DIR, IMAGE, SOURCE_LIST
from src.app.utils import load_model


def streamlit_app():
    # Setting page layout
    st.set_page_config(
        page_title="Разработка методики определения дефектов пиломатериалов на изображениях с использованием глубокого обучения",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Main page heading
    st.title("Определение дефектов пиломатериалов на изображениях")

    # Sidebar
    st.sidebar.header("Настройки модели")
    model_s = 0
    model_type = st.sidebar.radio(
        "Выбрать модель", ['Yolov8', 'Yolact', 'DeepLab'])
    confidence = float(st.sidebar.slider(
        "Change Confidence", 0, 100, 20)) / 100
    if model_type == "Yolov8":
        model_s = 0
        model_path = Path(WEIGHTS_DIR / 'best.pt')
    elif model_type == "Yolact":
        model_s = 0  # временно беру другую модель на выходных исправлю
        model_path = Path(
            WEIGHTS_DIR / 'yolact_plus_resnet50_defect_74_100000.pt')

    # Load Pre-trained ML Model
    try:
        model = load_model(model_path, model_s)

    except Exception as ex:
        st.error(
            f"Ошибка загрузки модели. Неправильный путь: {model_path}")
        st.error(ex)

    # Sidebar
    st.sidebar.header("Настройка данных")
    source_radio = st.sidebar.radio("Выбрать исходник", SOURCE_LIST)

    if source_radio == IMAGE:
        image_detector = YOLODetector(model, confidence)
        image_detector.detect()
