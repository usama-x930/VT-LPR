# Vehicle and License Plate Recognition with Novel Dataset for Toll Collection
This repository is the implementation of Vehicle and License Plate Recognition with Novel Dataset for Toll Collection (VT-LPR).

## Introduction
Tolling efficiency in manual toll collection system is low and time consuming. This requires human efforts and resources. Toll collection process is automated in some countries by installing sensors and Radio Frequency Identification (RFID) based system, but this comes with an additional cost of installing such systems. Utilizing the already installed cameras for survillence purposes, we automate the toll collection process by recognizing vehicle type and license plate from the image taken by the cameras.
We gather a Novel Vehicle type and License Plate Recognition Dataset called _Diverse Vehicle and License Plates Dataset (DVLPD)_ consisting of 10k images. We present an automated toll collection process which consists of three steps: Vehicle Type Recognition, License Plate Detection and Character Recognition. We train different state-of-the-art object detection models such as YOLO V2, YOLO V3, YOLO V4 and Faster RCNN. For the real-time application, we deploy our models on Raspberry Pi.

## Requirements
* Python>=3.6
* Streamlit==1.0.0
* PyQt5==5.15.4
* PyQt5Designer==5.14.1
* OpenCV==4.5.1.48
* pillow
* matplotlib
* pandas
* numpy
* tensorflow

Note: These packages can be installed through Pip package installer. This code is tested with these versions.
