import PIL
from constants import DEFAULT_IMAGE
import streamlit as st
from src.app.utils import sum_detections
from types import SimpleNamespace


def get_args(score_threshold, top_k):
    args_dict = {
        'crop': True,
        'score_threshold': score_threshold,
        'display_lincomb': False,
        'top_k': top_k,
        'eval_mask_branch': True,
    }
    return SimpleNamespace(**args_dict)


class YOLODetector:
    def __init__(self, model, accuracy,  width=1280, height=512, model_type=0):
        self.model = model
        self.accuracy = accuracy
        self.args = get_args(accuracy, 15)
        self.width = width
        self.height = height
        self.model_type = model_type

    def detect(self):
        image_process = None
        source_image = st.sidebar.file_uploader(
            "Выберите изоборажение", type=("jpg", "jpeg", "png")
        )
        col1, col2 = st.columns(2)
        with col1:
            try:
                if source_image is not None:
                    image_process = PIL.Image.open(source_image)
                    st.image(
                        image_process, caption="Исходное изображение", use_column_width=True)
                else:
                    default_image = PIL.Image.open(DEFAULT_IMAGE)
                    st.image(default_image, caption='Базовое изображение',
                             use_column_width=True)
                    image_process = default_image
            except Exception as ex:
                st.error(f"Error loading image")
                st.error(ex)

            if st.sidebar.button("Анализ (Предикт)"):
                if self.model_type == 0:
                    detected_objects_summary_list = []
                    res = self.model.predict(image_process, conf=self.accuracy)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    detected_objects_summary_list.extend(res[0].boxes.cls)
                    with col2:
                        st.image(res_plotted, caption='Detected Image',
                                 use_column_width=True)
                        try:
                            with st.expander("Результат"):
                                if not boxes:
                                    st.write(
                                        "Нет обнаруженных объектов(классов)")
                                else:
                                    for box in boxes:
                                        st.write(box.xywh)
                        except Exception as ex:
                            st.write("An error occurred while procesing")
                    if boxes:
                        sum_detections(
                            detected_objects_summary_list, self.model)
                # elif self.model_type == 1:
                #     detected_objects_summary_list = []
                #     self.model.eval()
                #     preds = eval.evalimage(self.model, image_process)
                #     preds = postprocess(
                #         preds, self.width, self.height,
                #         visualize_lincomb=self.args.display_lincomb,
                #         crop_masks=self.args.crop,
                #         score_threshold=self.args.score_threshold
                #     )
                #     idx = preds[1].argsort(0, descending=True)[
                #         :self.args.top_k]
                #     masks = preds[3][idx]
                #     masks = masks.detach().cpu().numpy()
                #     masks = [(mask * 255).astype('uint8') for mask in masks]
                #     #  return [InstanceSegmentationDefect(get_contours(mask)[0]) for mask in masks]
                #     res = [get_contours(mask)[0] from mask in masks]
