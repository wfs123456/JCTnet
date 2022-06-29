# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2022/1/18 17:24
# @File     : vis_grad_cam.py
# @Software : PyCharm

import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from Networks.Swin import SwinIR

class ShowGradCam:
    def __init__(self,conv_layer):
        assert isinstance(conv_layer,torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self,module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self,input_img):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img,str):
            input_img = cv2.imread(input_img)
        img_size = (input_img.shape[1],input_img.shape[0])
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        cam = self.gen_cam(fmap, grads_val)
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)/255.
        cam = heatmap + np.float32(input_img/255.)
        cam = cam / np.max(cam)*255
        cv2.imwrite('vis/img-3-000087.png',cam, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print('save gradcam result in grad_feature.png')

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_input = img_transform(img, transform)
    return img_input


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
    device = torch.device('cuda')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join("/home/hdd/dataset_xzw/TRANCOS_Nor/test/image-3-000087.jpg")
    model_path = os.path.join('/home/xuzhiwen/home/xuzhiwen/dataset_wfs/TransDensityNet/ckpts/VGG1-8-PartA+SwinIR/12-29-input-256-lr-1e-05-Count/best_model_mae 62.20.pth')

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img).to(device)
    model = SwinIR.Net()
    model.to(device)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    gradCam = ShowGradCam(model.conv_before_upsample) #............................. def which layer to show

    # forward
    output = model(img_input)
    print(output.size(), torch.sum(output))

    # backward
    model.zero_grad()
    class_loss = torch.sum(output)
    class_loss.backward()

    # save result
    gradCam.show_on_img(img) #.......................... show gradcam on target pic