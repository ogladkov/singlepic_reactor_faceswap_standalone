import hashlib
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid
import wget
import zipfile


def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def get_image_md5hash(image: Image.Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()


def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):

    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def download_ckpts(cfg):

    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    os.makedirs(cfg.models.ckpts_dir, exist_ok=True)

    try:

        if not os.path.exists(cfg.models.face_restoration_model.path):
            print('Downloading face restoration model...')
            wget.download(cfg.models.face_restoration_model.link, cfg.models.ckpts_dir, bar=bar_progress)
        print('\nRestoration model is read\n')

        buffalo_l_dir = os.path.join(cfg.models.insightface.home, 'models')

        if not os.path.exists(buffalo_l_dir):
            os.makedirs(buffalo_l_dir, exist_ok=True)
            print('Downloading Insightface models...')
            wget.download(cfg.models.insightface.link, buffalo_l_dir, bar=bar_progress)

        buffalo_l_zip = os.path.join(buffalo_l_dir, 'buffalo_l.zip')
        if os.path.exists(buffalo_l_zip):

            with zipfile.ZipFile(buffalo_l_zip, 'r') as zip_ref:
                buffalo_l_dir_out = os.path.join(buffalo_l_dir, 'buffalo_l')
                zip_ref.extractall(buffalo_l_dir_out)

            os.remove(buffalo_l_zip)
        print('\nInsightface model is read\n')

        if not os.path.exists(cfg.models.face_detection_model.path):
            print('Downloading face detection model...')
            wget.download(cfg.models.face_detection_model.link, cfg.models.ckpts_dir, bar=bar_progress)
        print('\nDetection model is read\n')

        if not os.path.exists(cfg.models.face_parse_model.path):
            print('Downloading face parsing model...')
            wget.download(cfg.models.face_parse_model.link, cfg.models.ckpts_dir, bar=bar_progress)
        print('\nFace parsing model is read\n')

        if not os.path.exists(cfg.models.face_swap_model.path):
            print('Downloading face swap model...')
            wget.download(cfg.models.face_swap_model.link, cfg.models.ckpts_dir, bar=bar_progress)
        print('\nFaceswap model is read\n')

    except Exception as e:
        print(f'An error appeared: {e}')
