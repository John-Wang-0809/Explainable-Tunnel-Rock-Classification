import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os
import numpy as np
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.xgrad_cam import XGradCAM

class yolov8_target(torch.nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

    def forward(self, data):
        logits = data[0]
        if logits.dim() == 0:  # If logits is a scalar
            logits = logits.unsqueeze(0)  # Convert scalar to tensor
        return logits


class yolov8_heatmap:
    def __init__(self, weight, device, method, layers, conf_threshold, renormalize):
        device = torch.device(device)
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(conf_threshold)
        self.model = model
        self.method_class = method
        self.target = target
        self.device = device
        self.renormalize = renormalize
        self.layers = layers

    def process(self, img_path, save_path, layer_idx, image_index):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # Generate Grad-CAM for specific layers
        target_layers = [self.model.model[layer_idx]]
        method = eval(self.method_class)(self.model, target_layers)
        grayscale_cam = method(tensor, targets=[self.target])
        grayscale_cam = grayscale_cam[0, :]

        # Overlay heatmap on image
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=0.5)
        cam_image = Image.fromarray(cam_image)

        # Save image with image index and layer number in filename
        cam_image.save(f'{save_path}/image{image_index}_layer{layer_idx}_{os.path.basename(img_path)}')

    def __call__(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            img_files = os.listdir(img_path)
            for index, img_file in enumerate(img_files, start=1):  # Image index starts from 1
                img_full_path = f'{img_path}/{img_file}'
                for layer_idx in self.layers:  # Iterate through each layer
                    self.process(img_full_path, save_path, layer_idx, index)
        else:
            for layer_idx in self.layers:  # 遍历每一层
                self.process(img_path, save_path, layer_idx, 1)  # Assume single image with index 1


def get_params():
    params = {
        'weight': 'runs/class/train=2/ROCKclass100+lossv3/weights/best.pt',
        'device': 'cuda:0',
        'method': 'XGradCAM',
        'layers': list(range(11)),  # All layers from 0 to 11
        'conf_threshold': 0.01,
        'renormalize': True
    }
    return params


if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'Datasets/data/Rock/image/grad1/', 'Xresult')
