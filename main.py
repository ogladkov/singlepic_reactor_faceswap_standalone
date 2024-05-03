from omegaconf import OmegaConf
from PIL import Image

from components.perform_faceswap import FaceSwapScript
from utils import download_ckpts


class FaceSwapper:

    def __init__(self, config_path: str) -> None:
        self.cfg = OmegaConf.load(config_path)
        download_ckpts(self.cfg)

    def execute(self, input_image_path: str, source_image_path: str) -> Image.Image:
        script = FaceSwapScript()
        source_img = Image.open(source_image_path)
        input_img = Image.open(input_image_path)
        swapped = script.process(source_img=source_img, input_img=input_img, cfg=self.cfg)

        swapped.show()

if __name__ == '__main__':
    source_image_path = '/home/sm00th/Projects/upwork/dayo/src/woman_6_1024.jpg'
    input_image_path = '/home/sm00th/Projects/upwork/dayo/src/portrait6_1024.jpg'
    # source_image_path = '/home/sm00th/Projects/upwork/dayo/src/portrait10.png'

    reactor = FaceSwapper('./config.yml')

    reactor.execute(input_image_path=input_image_path, source_image_path=source_image_path)
