# Tunnelling and Underground Space Technology + Supplementary Code

## Project Introduction

Based on a self-designed backbone network, this project introduces the following innovative improvements for the rock image classification task:

- **Introduction of Soft Token Shift Block (self-developed module)** to enhance feature representation capability.
- **Integration of Cross-Entropy Loss and Label Smoothing Loss** to improve model generalization.
- **Multi-layer attention heatmap visualization**, supporting analysis of attention distribution at each layer of the model.

## Environment Requirements

- Python >= 3.8

## Dataset Preparation

1. Trial link for the dataset used in the paper:
2. Place the dataset in the `datasets/` directory, or specify the path in the configuration file.
3. If you need to use your own dataset, you can follow the format of the trial dataset provided in this project.

## Usage

1. **Train the Model**

   ```bash
   python main.py  
   ```
   To further modify parameters, you can specify them in main.py:
   ```bash
    model.train(data="Datasets/data/Rock/image", batch=32,
                epochs=2, project='runs/class/train=2', name='ROCKclass',
                amp=False,
                workers=1,
                optimizer='AdamW',  # Optimizer
                # cos_lr=True,  # Cosine LR Scheduler
                #lr0=0.0005,
                lr0=0.001
                #imgsz=480
                )  # train
   ```

2. **Testing/Evaluation**

   - After training, you can perform inference and evaluation with the following code:
    ```bash
    model = YOLO(r"runs/class/train=2/ROCKclass3/weights/best.pt")
    model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", ch=3, batch=4, workers=1, save_json=True, save_txt=True)  # Validation
    model.predict(source=r"Datasets/data/Rock/test", save=True)  # Detection
    ```

3. **Visualization Analysis**
    

   ```bash
   python heatmap_cls.py  # for single-layer use
   python heatmap_cls_multi_layer.py  # for multi-layer use
   ```
    - You can scroll to the end of the above files to select the corresponding method and path:
   ```bash
   if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'Datasets/data/Rock/image/grad/', 'Xresult')
   ```
    - In addition, you can also select the method and parameters in yolov8_heatmap.py:
   ```bash
   def get_params(): # line 162
    params = {
        'weight': r'runs/detect/train=2/FINALCHECK/weights/best.pt',
        'cfg': r'ultralytics/cfg/models/v8/yolov8-twoCSP-CTF-CFE.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[9]',
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.6, # 0.6
        'ratio': 0.02 # 0.02-0.1
    }
    return params
    ```
   - After running the above code, the model will automatically generate various visualization results. Here are standard examples:
     - `output.jpg`: Standard GradCAM multi-layer visualization
     - `output++.jpg`: Improved GradCAM++ multi-layer visualization
     - `xoutput.jpg`: XGradCAM multi-layer visualization method




## Acknowledgements

- Contributors to related open-source projects and datasets 