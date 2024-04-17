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

### Run client (Запуск клиентской части)
```
streamlit run startup.py
```

### Developer
Балакишиев Валерий (Balakishiev Valery)



