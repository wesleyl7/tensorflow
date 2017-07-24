# TensorFlow for Absolute Beginners

This repository contains a few TensorFlow instructions and utility Python codes for absolute beginner.

Environment setup:
- MacBook Pro, NVIDIA Tegra X1
- virtualenv for Kivy development

## TensorFlow Installation

- Follow the instructions from https://www.tensorflow.org/install/ for your environment.
- For virtualenv or python environment, simply run "pip install tensorflow" should work.
- If you are using kivy like me, you should run "kivy -m pip install tensorflow"
- If you are running from Mac, please note "As of version 1.2, TensorFlow no longer provides GPU support on Mac OS X.", according to https://www.tensorflow.org/install/install_mac.

## TensorFlow Model Installation

- Before running any Tensorflow application using Tensorflow Models, you have to install the modles manually. Please following the instructions here: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md.
- The key is to run the "protoc" to compile the Protobuf libraries since Protobufs is used to configure module and training parameters.
- Another thing is to add the tensorflow/models and slim directory to your PYTHONPATH.

## TensorFlow Utilities

### Object Detection

The object detection source codes can be found in the ObjectDetection/tf_object_detect.py.

### Usage:

```python
obj_detector = tfImageObjectDetect()
obj_detector.image_object_detect(source_image, destination_image)
```
where
source_image is the source image file path, and
destination_image is the generated image file containing the boxes surrounding the detected objects.

### output

The output generated from the codes are:
```python
[INFO   ] Start object detection for image ./media/2039376.JPG with size (1280, 960)
(960, 1280, 3)
[INFO   ] [Finished object detection in 0]00:04.642936!!!
```

### Performance:
Detecting one JPEG image with size 1280x960 on the late 2013 MacBook Pro using CPU only is about 4.5~5 seconds.

Will try it on NVIDIA T-X1.
