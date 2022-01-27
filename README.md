# Helmet detection on Yolov5 using Jetson Nano 2gb
## Aim And Objectives
Aim

    To create a Helmet detection system which will detect Human head and then check if Helmet is worn or not.

Objectives

    • The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.
    • Using appropriate datasets for recognizing and interpreting data using machine learning.
    • To show on the optical viewfinder of the camera module whether a person is wearing a helmet or not.
## Abstract
    • A person’s head is classified whether a helmet is worn or not and is detected by the live feed from the system’s camera.
    • We have completed this project on jetson nano which is a very small computational device.
    • A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.
    • One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.
    • A helmet is by far the most important and effective piece of protective equipment a person on a motorcycle can wear. Helmets save lives by reducing the extent of head injuries in the event of a traffic accidents.
    • In a personal injury action brought by an injured motorcyclist, the opposing motorist may raise an issue regarding the motorcyclist's own negligence.


## Introduction
    • This project is based on a Helmet detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.
    • This project can also be used to gather information about who is wearing a helmet and who is not.
    • Helmets worn can further be classified into Safety helmets for Construction workers, helmets for Bicycle riders and helmets for Motorcycle riders based on the image annotation we give in roboflow. 
    • Helmet detection sometimes become difficult as safety helmets which don’t cover up chin and caps worn by people are of the same shape and size also sometimes the helmets worn by motorcyclists comes in variety of shapes and sizes with features like chin guard able to freely move above the head part thereby making helmet detection difficult. However, training in Roboflow has allowed us to crop images and change the contrast of certain images to match the time of day for better recognition by the model.
    • Neural networks and machine learning have been used for these tasks and have obtained good results.
    • Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Helmet detection as well.

## Literature Review
    • Wearing a helmet helps to reduce the impact of an accident on your head. While riding your two-wheeler, it is very likely that if you are involved in an accident, then the resulting head injuries can be fatal, if you are not wearing a helmet. 
    • A full-faced helmet covers your entire face, providing you the complete protection in case you go through an accident. This type of helmet protects your eyes from dust and high beam lights when driving your two-wheeler.
    • It is observed that wearing a helmet improves your attention while riding your bike. You tend to be more cautious and control your speed when wearing a helmet while driving your two-wheeler. 
    • Wearing a helmet not only covers your head but also covers your ears. This layer of safety blocks the cool breeze and rainy water to enter your ears and thus helps you to stay healthy & prevent you from getting sick in the cold and rainy weather.
    • The significance of a helmet is all the more germane as the roads are heavily flooded with speeding vehicles and the chances of being in an accident are very high. Also, the patch works, and continuous development of the Indian roads increase the chances of accidents.
    • A motorcyclist's legal recovery might be barred, or reduced, as a result of his/her contributory negligence in causing the accident.
    • Mandatory helmet laws for motorcycle operators and their passengers have, for the most part, proven to be an effective strategy in both increasing helmet use and reducing head injuries and fatalities in motorcycle accidents nationwide. But, while having an unmistakably positive effect on the overall safety of motorcycle riding, helmet laws have been met by resistance in the motorcycling community.

## Jetson Nano Compatibility
    • The power of modern AI is now available for makers, learners, and embedded developers everywhere.
    • NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.
    • Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.
    • NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    • In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.
## Jetson Nano 2gb

![Jetson Nano](https://github.com/danypetkar/Helmet-detection-using-Jetson-Nano/blob/main/IMG_20220125_115121.jpg)


## Proposed System
1] Study basics of machine learning and image recognition.

2]Start with implementation
    
    • Front-end development
    • Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Helmet or not.

4] use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a helmet or not.
## Methodology

The Helmet detection system is a program that focuses on implementing real time Helmet detection.
It is a prototype of a new product that comprises of the main module:
Helmet detection and then showing on viewfinder whether the person is wearing a helmet or not.
Helmet Detection Module

This Module is divided into two parts:


1] Head detection


    • Ability to detect the location of a person’s head in any input image or frame. The output is the bounding box coordinates on the detected head of a person.
    • For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
    • This Datasets identifies person’s head in a Bitmap graphic object and returns the bounding box image with annotation of Helmet or no Helmet present in each image.
2] Helmet Detection


    • Recognition of the head and whether Helmet is worn or not.
    • Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.
    • There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
    • YOLOv5 was used to train and test our model for whether the helmet was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%. 



## Installation

Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

```
Create Swap 
```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```
Cuda env in bashrc
```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

```
Update & Upgrade
```bash
sudo apt-get update
sudo apt-get upgrade
```
Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
Clone Yolov5 
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
## Helmet Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder
link of project

Insert gif or link to demo


## Running Helmet Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo
[![Watch the video]](https://github.com/danypetkar/Helmet-detection-using-Jetson-Nano/blob/main/helmet_output.mp4)
## Advantages

    • Helmet detection system will be of great help in minimizing the injuries that occur due to an accident.
    • Helmet detection system shows whether the person in viewfinder of camera module is wearing a Helmet or not with good accuracy.
    • It can then convey it to authorities like traffic policeman or the data about the respective person and his vehicle can be stored, and then based on the info acquired can be notified on his mobile phone about the Helmet using law.
    • When completely automated no user input is required and therefore works with absolute efficiency and speed.
    • It can work around the clock and therefore becomes more cost efficient.
## Application

    • Detects a person’s head and then checks whether Helmet is worn or not in each image frame or viewfinder using a camera module.
    • Can be used anywhere where traffic lights are installed as their people usually stop on red lights and Helmet detection becomes even more accurate.
    • Can be used as a reference for other ai models based on Helmet Detection.
## Future Scope

    • As we know technology is marching towards automation, so this project is one of the step towards automation.
    • Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
    • Helmet detection will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.

## Conclusion

    • In this project our model is trying to detect a person’s head and then showing it on viewfinder, live as to whether Helmet is worn or not as we have specified in Roboflow.
    • The model tries to solve the problem of severe head injuries that occur due to accidents and thus protects a person’s life.
    • The model is efficient and highly accurate and hence reduces the workforce required.
## Reference

1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=helmet

3] Google images
## Articles :-

1] https://www.bajajallianz.com/blog/motor-insurance-articles/what-is-the-importance-of-wearing-a-helmet-while-riding-your-two-wheeler.html#:~:text=Helmet%20is%20effective%20in%20reducing,are%20not%20wearing%20a%20helmet.

2] https://www.findlaw.com/injury/car-accidents/helmet-laws-and-motorcycle-accident-cases.html
