import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class yolov8_target(torch.nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

    def forward(self, data):
        logits = data[0]
        print(f"Logits shape: {logits.shape}")
        if logits.dim() == 0:  # If logits is a scalar
            logits = logits.unsqueeze(0)  # Convert scalar to tensor

        return logits


class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, conf_threshold, renormalize):
        device = torch.device(device)
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(conf_threshold)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')

        self.model = model
        self.method = method
        self.target = target
        self.device = device
        self.renormalize = renormalize

    def process(self, img_path, save_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        grayscale_cam = self.method(tensor, targets=[self.target])
        grayscale_cam = grayscale_cam[0, :]

        # Adjust heatmap transparency to make background clearer
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=0.5)

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path, image_name):
        #  if os.path.exists(save_path):
           # shutil.rmtree(save_path)
        #  os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/{image_name}.png')


def get_params():
    params = {
        'weight': 'runs/classify/train16/weights/best.pt',
        'device': 'cuda:0',
        'method': 'GradCAM',
        'layer': [0],
        'conf_threshold': 0.01,
        'renormalize': True
    }
    return params


if __name__ == '__main__':
      model = yolov8_heatmap(**get_params())
      model(r'Datasets/data/Rock/test/', 'result', 'result')
# if __name__ == '__main__':
#     # Initialize model
#     model = yolov8_heatmap(**get_params())
#
#     # 图像名称列表，作为一个字符串变量
#     image_names = """
#    0c6c14d9f9eef9831f21c3d385903d3
#     """
#
#     # 将字符串拆分为列表
#     image_names_list = image_names.split()
#
#     # 遍历图像名称列表
#     for image_name in image_names_list:
#         # 构造完整的图像路径
#         image_path = f'Datasets/data/Rock/val/3/{image_name}.jpg'
#
#         # 调用模型进行处理并保存结果
#         model(image_path, 'result', image_name)
