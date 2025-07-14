from ultralytics import YOLO
import torch
if __name__ == '__main__':
    '''
    
    print(torch.cuda.is_available())
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Device:", device)
    print("GPU型号： ", torch.cuda.get_device_name(0))
    '''
    import os

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    ############## This is the training code ##############
    model = YOLO(r"ultralytics/cfg/models/v8/yolov8-cls.yaml")  # Initialize model
    #model = YOLO(r"ultralytics/cfg/models/v8/yolov8-early.yaml")  # Initialize model
    #model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP.yaml")  # Initialize model
    #odel = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF.yaml")  # Initialize model
    #model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF-CFE.yaml")  # Initialize model


    
    model.train(data="Datasets/data/Rock/image", batch=32,
                epochs=2, project='runs/class/train=2', name='ROCKclass',
                amp=False,
                workers=1,
                optimizer='AdamW',  # Optimizer
                # cos_lr=True,  # Cosine LR Scheduler
                #lr0=0.0005,
                lr0=0.001
                #imgsz=480
                )  # Training

    ############## This is the validation and prediction code ##############
    #model = YOLO(r"runs/class/train=2/ROCKclass3/weights/best.pt")
    #model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", ch=3,batch=4,  workers=1, save_json=True, save_txt=True)  # Validation
    #model.predict(source=r"Datasets/data/Rock/test", save=True)  #   Detection
