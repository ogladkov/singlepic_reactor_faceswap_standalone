import torch

from .parsenet import ParseNet


def init_parsing_model(cfg, device='cpu'):
    model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    load_net = torch.load(cfg.models.face_parse_model.path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
