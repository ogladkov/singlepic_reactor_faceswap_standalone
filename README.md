# REACTOR STANDALONE APPLICATION

## Installation
To install run `pip install -e .` inside cloned project

## Configuration
- providers: list of providers to use in onnxruntime (set only CPUExecutionProvider if you want to use only CPU)
- models: setting (name, path and link for dolwnloading) for each model
- params: face restoration settings:
  - face_restore_visibility: strength of face restoration (minimum -> soapy, maximum -> crispy)
  - codeformer_weight: strength of CodeFormer (minimum -> closer to input face, maximum -> closer to swap face)

## Running
### In code
* Import **FaceSwapper** class from **reactor_standalone** to your code
* Feed the instance with a **config.yml** path
* Execute **.do_swap()** method

### Standalone
* Run `python main.py --source_image_path <SOURCE_IMAGE> --input_image_path <DESTINATION_IMAGE>`
* You can also use --config flag to specify a path to the config file

## Example
```python
from reactor_standalone import FaceSwapper

# Input images
source_image_path = './test_imgs/w_01.png'
input_image_path = './test_imgs/w_02.png'

# Utilization
fsw = FaceSwapper('./config.yml')
swapped = fsw.do_swap(input_image_path=input_image_path, source_image_path=source_image_path)
```