import argparse

from . import FaceSwapper

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
