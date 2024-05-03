from copy import deepcopy

import torch

from .retinaface import RetinaFace


def init_detection_model(cfg, half=False, device='cuda'):
    return init_retinaface_model(cfg, half, device)


def init_retinaface_model(cfg, half=False, device='cuda'):
    model = RetinaFace(network_name='resnet50', half=half)
    load_net = torch.load(cfg.models.face_detection_model.path, map_location=lambda storage, loc: storage)

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():

        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)

    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model