# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:26:58 2020

@author: Mao Jianqiao
"""
import cv2
from sklearn.externals import joblib
import numpy as np
from keras.models import load_model

def contLowEdge(li,contNum):
    zeroCount=0
    oneCount=0
    le=9999
    for i in range(len(li)):
        if li[i]==0:
            zeroCount+=1
            if zeroCount>=contNum:
                le=i
            oneCount=0
        else:
            oneCount+=1
            if oneCount<80:
                None
            else:
                le=9999
                break
            zeroCount=0
    return le

def contUpEdge(li,contNum):
    zeroCount=0
    oneCount=0
    locker=0
    le=9999
    for i in range(len(li)):
        if li[i]==0:
            zeroCount+=1
            if zeroCount==contNum and locker==0:
                locker=1
                le=i-contNum+1
            oneCount=0
        else:
            oneCount+=1
            if oneCount<80:
                None
            else:
                le=9999
                break
            zeroCount=0
    return le

def Cropping(frame):
    
    row, col= frame.shape #row = 600， col=800
    ##### Remove noise on the top and bottom edge  #####
    effPtNum1=np.zeros((1, row), dtype='uint16')
    judge1 = np.zeros((1, row), dtype='uint16') 
    
    for j in range(row): #row j from 0 to 599.
        effL= np.array(np.nonzero(frame[j,:int((col/2))]))
        effR=np.array(np.nonzero(frame[j,int(col/2):]))+int(col/2)
        ## measure the distance between two inner points
        if effR.shape[1]!=0 and effL.shape[1]!=0:    
            dist=effR[0,0]-effL[0,-1]
            if dist>=0.85*col: ## if there is a col that two inner points separating from each more than 0.85 times length of the col
                frame[j,:]=0
        effPtNum1[0,j]=effR.shape[1]+effL.shape[1]
    
    max_num1 = max(effPtNum1[0, :]) #Calculate the maxima of the number of effective points in a single row.
    threshold1 = max_num1 * 0.05 #set the threshold using to filter the row 
    ##Judge each row one by one. If the number of effective points is larger than threshold, record 1 in judge1.
    for j in range(row): 
        if (effPtNum1[0,j] >= threshold1):
            judge1[0, j]=1
        else:
            None
    
    ##Calculate the upper and nether bound of effective region
    #Find bound at the top, and then clean the edge region
    topEdge=contLowEdge(judge1[0,int(0.1*row)-10:int(0.4*row)+10],20)+int(0.1*row)-10
    topEpty=[i for i,ele in enumerate(judge1[0,:int(0.1*row)]) if ele==0]
    if topEdge==9999+int(0.1*row)-10: 
        if len(topEpty):
           topEdge = max(topEpty)
        else:
           topEdge=0
    else:
        None
    judge1[0,:topEdge]=0
    
    #Find bound at the bottom, and then clean the edge region
    bottomEdge=contUpEdge(judge1[0,int(0.6*row)-10:int(0.9*row)+10],20)+int(0.6*row)-10
    bottomEpty=[i for i, ele in enumerate(judge1[0,int(0.9*row):]) if ele==0]
    if bottomEdge==9999+int(0.6*row)-10:
        if len(bottomEpty):
            bottomEdge = min(bottomEpty)+int(0.9*row)
        else:
            bottomEdge=row
    else:
        None
    judge1[0,bottomEdge:]=0
    #Find the upper and nether cropping boundary
    effRow = np.array(np.nonzero(judge1))
    if len(effRow[1,:]):
        upCropping = effRow[1,0] #the upper edge is the first item of judge1
        lowCropping = np.max(effRow)  #the nether edge is the last item of judge1
    else:
        upCropping=0
        lowCropping=row
     
    ##### Remove noise on the left and right edge   #####
    effPtNum2=np.zeros((1, col), dtype='uint16')
    judge2 = np.zeros((1, col), dtype='uint16') #Create a 1*800 vector
    
    for i in range(col): #col m from 0 to 799.
        effU = np.array(np.nonzero(frame[:int((row/2)),i]))
        effD=np.array(np.nonzero(frame[int(row/2):,i]))+int(row/2)
        if effU.shape[1]!=0 and effD.shape[1]!=0:    
            dist2=effD[0,0]-effU[0,-1]
            if dist2>=0.7*row:
                frame[:,i]=0
        effPtNum2[0,i]=effU.shape[1]+effD.shape[1]
    max_num2 = max(effPtNum2[0, :]) #Calculate the maximum of the number of effective points in a single row.
    threshold2 = max_num2 * 0.08 #set the threshold
    ##Judge each col one by one. If the number of effective points is larger than threshold, record 1 in judge2.
    for i in range(col): 
        if (effPtNum2[0,i] >= threshold2):
            judge2[0, i]=1
        else:
            None
            
    ##Calculate the left and right bound of effective region
    #Find bound at the left, and then clean the edge region
    lfEdge=contLowEdge(judge2[0,int(0.1*col)-10:int(0.4*col)+10],20)+int(0.1*col)-10
    lfEpty=[i for i,ele in enumerate(judge2[0,:int(0.1*col)]) if ele==0]
    if lfEdge==9999+int(0.1*col)-10:
        if len(lfEpty):
            lfEdge = max(lfEpty)
        else:
            lfEdge=0
    else:
        None
    judge2[0,:lfEdge+1]=0
    
    #Find bound at the right, and then clean the edge region
    rtEdge=contUpEdge(judge2[0,int(0.6*col)-10:int(0.9*col)+10],20)+int(0.6*col)-10
    rtEpty=[i for i, ele in enumerate(judge2[0,int(0.9*col):]) if ele==0]
    if rtEdge==9999+int(0.6*col)-10 :
        if len(rtEpty):
            rtEdge = min(rtEpty)+int(0.9*col)
        else:
            rtEdge=col
    else:
        None
    judge2[0,rtEdge:]=0
    
    #Find the left and right cropping boundary
    effCol = np.array(np.nonzero(judge2))
    if len(effCol[1,:]):
        leftCropping = effCol[1,0] #the left edge is the first itme of judge2
        rightCropping = np.max(effCol) #the right edge is the last itme of judge2
    else:
        leftCropping=0
        rightCropping=col
    croppingEdge=[upCropping,row-lowCropping,leftCropping,col-rightCropping]
    cropping = frame[upCropping:lowCropping, leftCropping:rightCropping]
    return cropping, croppingEdge

def Segmentation(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerBd1 = np.array([0, 30 , 130])
    upperBd1 = np.array([15,100, 255])
    lowerBd2 = np.array([165,30, 130])
    upperBd2 = np.array([180,100, 255])
    mask1 = cv2.inRange(HSV, lowerBd1, upperBd1)
    mask2 = cv2.inRange(HSV, lowerBd2, upperBd2)
    mask=mask1+mask2
    return mask

def Scaling (frame):
    scale = cv2.resize(frame, (32, 32), cv2.INTER_LINEAR)
    return scale

alpha_12=['A','B','C','D','E','F','G','H','L','U','J','Z']
print("Welcome to use MJQ's visiton-based hand gesture recognition system (ASL version).", end='\n')
print("Model files loading...")
lstm=load_model('preclassifier_lstm.h5')
knn = joblib.load('static_classifier_knn.pkl')
PCA_Model= joblib.load('PCA.m')
hmm_J = joblib.load('dynamic_classifier_hmm_J.pkl')
hmm_Z = joblib.load('dynamic_classifier_hmm_Z.pkl')
print("Model files loading succeeds!",end='\n')

print("Camera initializing...",end='\n')
cap = cv2.VideoCapture(1)
cap.set(3, 800)
cap.set(4, 1200)
print("Camera initializing succeeds!",end='\n')

print("Recognitionn process start!",end='\n')
print('Please press "s" on the keyboard to start frame capturing.',end='\n')
while(True):
    ret, frame = cap.read()
    display = cv2.resize(frame,(1200,800))
    cv2.namedWindow('display', cv2.WINDOW_FREERATIO)
    cv2.imshow('display', display) #Display the sampling points as white points.

## frame capturing ##
    if cv2.waitKey(1) & 0xFF == ord('s'): 
        print("Start sampling! 10 frames will be captured.")
        print('Please press "c" one by one to capture each frame.')
        frameset=[]
        frameNo=0
        while True:        
            ret, frame = cap.read()
            display = cv2.resize(frame,(1200,800))
            cv2.namedWindow('display', cv2.WINDOW_FREERATIO)
            cv2.imshow('display', display)
            if cv2.waitKey(1) & 0xFF == ord('c'): 
                frameset.append(frame)
                frameNo+=1
                print('\r frame No. %d has been recorded.' %(frameNo),end='')
                if frameNo==10:
                    break

## pre-processing ##    
        print("\n Captured frames processing...",end='\n')
        for i in range(10):
            mask=Segmentation(frameset[i])
            cropping1,E1=Cropping(mask)
            cropping2,E2=Cropping(cropping1)
            cropping3,E3=Cropping(cropping2)
            BF=cropping3.shape
            E=np.array(E1)+np.array(E2)+np.array(E3)
            retImg=np.zeros((600,800))
            retImg[E[0]:600-E[1],E[2]:800-E[3]]=cropping3
            scale_WithoutBF = Scaling(retImg)
            scale_WithBF = Scaling(cropping3)
            vect_WithoutBF=scale_WithoutBF.flatten()
            vect_WithBF=np.hstack((np.array(scale_WithBF.flatten()),BF))            
            if i==0:
                RoI_series_WithoutBF=vect_WithoutBF
                RoI_series_WithBF=vect_WithBF
            else:
                RoI_series_WithoutBF=np.hstack((RoI_series_WithoutBF,vect_WithoutBF))
                RoI_series_WithBF=np.hstack((RoI_series_WithBF,vect_WithBF))
        normMinMax= joblib.load('Normalization.pkl')
        NormRoI_series_WithoutBF = (RoI_series_WithoutBF.reshape(-1,1024)/255.0).reshape(1,-1)
        NormRoI_series_WithBF = np.array(normMinMax.transform(RoI_series_WithBF.reshape(-1,1026))).reshape(1,-1)
        print("Preprocessed data has been ready. Start pre-classification.",end='\n')

## pre-classification ##
        preClaPredict=np.argmax(lstm.predict(NormRoI_series_WithoutBF.reshape((-1,10,1024))))
        if preClaPredict==0:
            print('This captured gesture is recognized as a static gesture.',end='\n')
        else:
            print('This captured gesture is recognized as a dynamic gesture.',end='\n')

## Classification ##       
        if preClaPredict==0:    
            print('Principal features extracting...',end='\n')
            x=PCA_Model.transform(NormRoI_series_WithBF.reshape(-1,1026))
            print('50 principal features are extracted! Start classification.',end='\n')
            voteVector=list(knn.predict(x))
            pred=(max(voteVector, key=voteVector.count))
        else:
            print('Start classification.',end='\n')
            score_J= hmm_J.score(NormRoI_series_WithoutBF)
            score_Z= hmm_Z.score(NormRoI_series_WithoutBF)
            pred=np.argmax(np.array([score_J,score_Z]))+10
        predClass=alpha_12[int(pred)]
        print('The recognized gesture is "%s" in ASL alphabet' %(predClass)) 
