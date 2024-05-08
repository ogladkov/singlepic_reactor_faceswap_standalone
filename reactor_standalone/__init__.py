from omegaconf import OmegaConf
from PIL import Image

from .components.perform_faceswap import FaceSwapScript
from .utils import download_ckpts


class FaceSwapper:

    def __init__(self, config_path: str) -> None:
        self.cfg = OmegaConf.load(config_path)
        download_ckpts(self.cfg)

    def do_swap(self, input_image_path: str, source_image_path: str) -> Image.Image:
        script = FaceSwapScript()
        source_img = Image.open(source_image_path)
        input_img = Image.open(input_image_path)
        swapped = script.process(source_img=source_img, input_img=input_img, cfg=self.cfg)

        return swapped
