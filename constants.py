from pathlib import Path

# Path
ROOT_DIR = Path(__file__).resolve(strict=True).parent
SRC_DIR = ROOT_DIR / 'src'
WEIGHTS_DIR = ROOT_DIR / 'weights'
DEFAULT_IMAGE = ROOT_DIR / 'images' / 'test.jpg'
DATA_DIR = ROOT_DIR / 'data' / 'dataset_yolo'


# Source
IMAGE = 'Image'
VIDEO = 'Video'
SOURCE_LIST = [IMAGE, VIDEO]

# List model YOLO (YAML, PT)
MODEL_YAML_N = 'yolov8n-seg.yaml'
MODEL_SEG_N = 'yolov8n-seg.pt'
MODEL_YAML_S = 'yolov8s-seg.yaml'
MODEL_SEG_S = 'yolov8s-seg.pt'
MODEL_YAML_M = 'yolov8m-seg.yaml'
MODEL_SEG_M = 'yolov8m-seg.pt'
MODEL_YAML_L = 'yolov8l-seg.yaml'
MODEL_SEG_L = 'yolov8l-seg.pt'
MODEL_YAML_X = 'yolov8x-seg.yaml'
MODEL_SEG_X = 'yolov8x-seg.pt'
