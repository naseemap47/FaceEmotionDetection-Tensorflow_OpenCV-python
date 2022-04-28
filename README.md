# FaceEmotionDetection-Tensorflow_OpenCV-python

<p align="center">
 <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python/">
    <img alt="Python3" src="https://img.shields.io/badge/Language-Python3-yellowgreen?color=brightgreen&logo=python">
  </a>
  <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python//issues">
    <img alt="Top-down learning path: FaceEmotionDetection-Tensorflow_OpenCV-python" src="https://img.shields.io/github/issues/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python?color=9cf&style=flat&logo=appveyor">
  </a>
  <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python?color=success&style=flat&logo=appveyor">
  </a>
  <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python/network">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python?style=flat&logo=Git">
  </a>
  <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python/blob/master/LICENSE">
    <img alt="GitHub Licence" src="https://img.shields.io/github/license/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python?color=red&style=flat&logo=appveyor">
  </a>
  <a href="https://www.linkedin.com/in/naseem-alassampattil/">
    <img alt="Linkedin" src="https://img.shields.io/badge/Linkedin-blue?logo=linkedin">
  </a>
 <a href="https://github.com/naseemap47">
    <img alt="Github" src="https://img.shields.io/badge/Github-black?logo=github">
 </a>
 <a href="https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python">
    <img alt="pyCharm" src="https://img.shields.io/badge/IDE-pyCharm-yellowgreen?color=brightgreen&logo=pycharm">
  </a>
</p>

## Description
Predict your emotions from Webcam using Tensorflow and OpenCV in python.

## ⚠️ Limitations
#### 1st Model
* **Model = 3-CNN layer + One Fully Connected Layer**
* Early Stopping at **36th** Epoch
* I got **54.95%** Accuracy, Val_accuracy = **52.75%**
#### 2nd Model
* **Model = 4-CNN layer + One Fully Connected Layer**
* For My system it's take **22 hr** to Complete the Processes (Early Stopping at **41th** Epoch)
* I got **68.59%** Accuracy, Val_accuracy = **66.28%**
#### 3rd Model
* Only **70.42%** Accuracy,  Val_accuracy = **63.40%**
* Trained using pre-trained model (**mobilenetv2_1.00_224**)
* For My system it's take **3.1 hr** to Complete the Training (Early Stopping at **30th** Epoch)


## 💡 Suggestions
* Train in system having **GPU** (Graphics Card)
   * eg:- Nvidia RTX 1080 
* Increase depth of neural networks

## Requirements
#### Programming Language
* Python3 (pyCharm)
#### Libraries
* OpenCV
* TensorFlow
* Keras
* Numpy
* Matplotlib
* OS

## Project Setup
* Clone the project
* RUN - **detect_face_emotions.py**
* Detect Face Emotions in Web-cam or in media file (eg: CCTV Footage)
  * If we want use web-cam or Media file (Change Value in VideoCapture)
  * Web-cam -> VideoCapture(0)
  * Video file -> VideoCapture("path_to_media_file")
* To **Quit** - Press **Q-key**

## License
[![CC0](http://seawisphunter.com/minibuffer/api/MIT-License-transparent.png)](https://github.com/naseemap47/FaceEmotionDetection-Tensorflow_OpenCV-python/blob/master/LICENSE)
