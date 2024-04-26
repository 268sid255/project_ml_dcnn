## Project on Samsung - Development of a methodology for identifying lumber defects in images using deep learning.<br/>(Проект на Самсунг - Разработка методики определения дефектов пиломатериалов на изображениях с использованием глубокого обучения)


### Structure of this Repo (Структура репозитория)
- [src](src) : main code
- [app](src/app) : client (клиенская часть)
- [yolact](src/yolact) : Yolact train
- [yolo_seg](src/yolo_seg) : Yolo train

### Training YOLO (обучение с использование YOLO)
#### Trains using the base args image size, epochs, batch_size (-1 run auto batch_size), iou, type_model_load (n,s,m,l or x, default load m)
```
python yolo_seg/train.py --imgsz=640 --epochs=50 --batch_size=8 --name='test_model' --iou=0.5 --type_model_load='m'
```
### Training YOLACT (обучение с использование YOLACT)

 - Для обучения возьмите предварительно обученную модель imagenet и поместите ее в папку `./weights`.
   - Для Resnet101 загрузите `resnet101_reducedfc.pth` из [здесь](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - Для Resnet50 загрузите `resnet50-19c8e357.pth` из [здесь](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
   - Для Darknet53 загрузите файл `darknet53.pth` с [здесь](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).
 - Запустите одну из команд обучения, приведенных ниже.
   - Обратите внимание, что во время обучения вы можете нажать ctrl+c, и это сохранит файл `*_interrupt.pth` на текущей итерации.
   - Все веса по умолчанию сохраняются в директории `./weights` с именем файла `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```
### Run client (Запуск клиентской части)
```
streamlit run startup.py
```
### Презентация
Находится в папке presentation
### Developer
Балакишиев Валерий (Balakishiev Valery)



