# Vision-based-Sign-Language-Recognition-System

## Overview:

<div align=center><img src=https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/blob/main/doc/SYSTEM%20FRAMEWORK.png width=900 /></div>

My undergraduate Final Year Project awarded as the Excellent Bachelor's Project. It develops a vision-based sign language recognition system with multiple machine-learning models, which currently can recognize 10 static and 2 dynamic gesutures in ASL with testing accuracy of 99.68%.

## Classifier implemented by:

  Pre-classifier: LSTM
  
  Static gesture classifier: KNN
  
  Dynamic gesture classifier: 2 HMMs

## Files introduction:

There are two .py file:

  **1) [Training_Test_Using_Prepared_Datasets.py](https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/blob/main/Training_Test_Using_Prepared_Datasets.py)**

  To train, test and save models. 
  Read two datasets with and without bounding features as input. Use 5-fold CV to train and test models. Evaluation metrics(accuracy, precision, recall and f1 score), confusion matrices are output. Models trained in the final round of 5-fold CV are saved for further usage.

  **2) [gesture_recognition_system.py](https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/blob/main/gesture_recognition_system.py)**
  
  To operate as the real-time recognition system with pre-trained models. Data acquisition, preprocessing, feature extraction, pre-classification and classification processes are included. Read the saved models generated by 1).

There are a [system framework diagram](https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/blob/main/doc/SYSTEM%20FRAMEWORK.png) and a [demo. video](https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/blob/main/doc/FYP%20demo.%20video.mp4) for reference.

[Model files folder](https://github.com/JianqiaoMao/Real-time-Vision-based-Sign-Language-Recognition-System/tree/main/model%20files) contains pre-trained models as an example.

## Datasets: 

Datasets are not available for public use at this stage.

## Camera: GUCEE HD98

Please note that different camera type can lead to significant variance in system performance.
