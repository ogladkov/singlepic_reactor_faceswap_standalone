import argparse

from omegaconf import OmegaConf
from PIL import Image

from components.perform_faceswap import FaceSwapScript
from utils import download_ckpts


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input image (destination) path', required=True, default='./test_imgs/w_01.png')
    parser.add_argument('--source', help='Source image path', required=True, default='./test_imgs/w_01.png')
    parser.add_argument('--config', help='Input file path', required=False, default='./config.yml')
    args = parser.parse_args()

    fsw = FaceSwapper(args.config)
    swapped = fsw.do_swap(input_image_path=args.input, source_image_path=args.source)
    swapped.show()


if __name__ == '__main__':
    main()
