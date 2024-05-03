# REACTOR STANDALONE APPLICATION

## Installation
* Setup your env (python 3.10)
* Run `sh ./install.sh` - it will install PyTorch with CUDA and the rest of the dependencies
* Weights of all neural networks will be downloaded automatically during the first run and will be placed at './checkpoints/' folder

## Configuration
- providers: list of providers to use in onnxruntime (set only CPUExecutionProvider if you want to use only CPU)
- models: setting (name, path and link for dolwnloading) for each model
- params: face restoration settings:
  - face_restore_visibility: strength of face restoration (minimum -> soapy, maximum -> crispy)
  - codeformer_weight: strength of CodeFormer (minimum -> closer to input face, maximum -> closer to swap face)

## Running
* Import FaceSwapper class from main.py to your code
* Feed the instance with a config.yml path
* Execute *do_swap* method

## Example
```python
# Input images
source_image_path = './test_imgs/w_01.png'
input_image_path = './test_imgs/w_02.png'

# Utilization
fsw = FaceSwapper('./config.yml')
swapped = fsw.do_swap(input_image_path=input_image_path, source_image_path=source_image_path)
```