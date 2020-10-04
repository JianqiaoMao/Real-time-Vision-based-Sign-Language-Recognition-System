# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:02:12 2020

@author: Mao Jianqiao
"""
from sklearn.decomposition import PCA 
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.externals import joblib
from hmmlearn import hmm
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#        

def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]
    row = table.nrows #Read the number of rows in excel.
    col = table.ncols #Read the number of columns in excel.
    datamatrix = np.zeros((row, col))  # Create a zero matrix with the same size as excel.
    for x in range(col):
        cols = np.matrix(table.col_values(x))
        datamatrix[:, x] = cols  # Collect data in excel into matrix.
    return datamatrix


def dig2let(indexList,gesNo):
    letList=[]
    alpha_5=['B','D','F','L','U']
    alpha_12=['A','B','C','D','E','F','G','H','L','U','J','Z']
    for i in range(len(indexList)):
        if gesNo==5:
            letList.append(alpha_5[indexList[i]])
        else:
            letList.append(alpha_12[indexList[i]])
    return letList

dataset_WithBF=excel_to_matrix('Final_Dataset_withBF.xlsx')
dataset_WithoutBF=excel_to_matrix('Final_Dataset_withoutBF.xlsx')
## Extract X, Y and Pre-Y from the dataset##
x_WithBF=dataset_WithBF[:,:10260]
x_st_WithBF=x_WithBF[:2000,:]
x_dy_WithBF=x_WithBF[2000:,:]

x_WithoutBF=dataset_WithoutBF[:,:10240].reshape(-1,1024)[:,:1024].reshape(-1,10240)
x_st_WithoutBF=x_WithoutBF[:2000,:]
x_dy_WithoutBF=x_WithoutBF[2000:,:]

y=dataset_WithBF[:,10260] # 0-11 for 12 classes of gesture
Pre_y=dataset_WithBF[:,10261] # 0 for static, 1 for dynamic


#parameter for KNN
k=4

#parameters for LSTM
nb_lstm_outputs = 20  #number of neuron
nb_time_steps = 10  #length of time series
nb_input_vector = 1024 #lengh of input in each time step

## 5-Fold cross validation
kf=KFold(5,True) 

Preacc=[]
Preprec=[]
Pref1score=[]
Prerecall=[]

acc=[]
prec=[]
f1score=[]
recall=[]

for train_index, test_index in kf.split(x_WithBF):
########## pre-classification ##########
## dataset separation for pre-classificaiton ##   
    Pre_x_train, Pre_x_test = x_WithoutBF[train_index], x_WithoutBF[test_index]
    Pre_y_train, Pre_y_test = Pre_y[train_index], Pre_y[test_index]
    PretrainSet=np.hstack((Pre_x_train,Pre_y_train.reshape(-1,1)))
    yy=PretrainSet[:,10240]
    Prex_trainSplit= np.array ( [PretrainSet[yy==i,:10240] for i in range(2)] )
    Prey_trainSplit= np.array ( [PretrainSet[yy==i,10240] for i in range(2)] )
    Pre_x_train=Pre_x_train.reshape((-1,10,1024))
    Pre_x_test=Pre_x_test.reshape((-1,10,1024))
    Pre_y_train = to_categorical(Pre_y_train, num_classes=2)
    Pre_y_test = to_categorical(Pre_y_test, num_classes=2)    

    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(Dense(2, activation='sigmoid'))
    
    #compile:loss, optimizer, metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(Pre_x_train, Pre_y_train, epochs=10, batch_size=4, verbose=1,validation_data = (Pre_x_test,Pre_y_test))
    PreClapredict = np.argmax(model.predict(Pre_x_test),axis=1)
    Pre_y_test=np.argmax(Pre_y_test,axis=1)
    Pre_y_train=np.argmax(Pre_y_train,axis=1)
## preclassifier metrics evaluation ##
    Preacc.append(accuracy_score(Pre_y_test, PreClapredict))
    Preprec.append(precision_score(Pre_y_test,PreClapredict))
    Pref1score.append(f1_score(Pre_y_test,PreClapredict))
    Prerecall.append(recall_score(Pre_y_test,PreClapredict))

     
########## classification ##########
#### dataset separation for classification ####     
    x_train, x_test = x_WithBF[train_index], x_WithBF[test_index]
    y_train, y_test = y[train_index], y[test_index]

   

####use the result of pre-classsification as dynamic ('1') or static ('0') lable
####and then split the dynamic and static set. 
## Training set preparation ###
    trainSet=np.hstack((x_train,y_train.reshape(-1,1)))
    trainSet=np.hstack((trainSet,Pre_y_train.reshape(-1,1)))
    
    train_Prelabel=trainSet[:,10261]
    train_label=trainSet[:,10260]
    
    # Split the Dynamic and Static samples for training (With Bounding Features)#    
    x_trainDyStSplit_WithBF= np.array ( [trainSet[train_Prelabel==i,:10260] for i in range(2)] )
    y_trainDyStSplit_WithBF= np.array ( [trainSet[train_Prelabel==i,10260] for i in range(2)] )

    trainSet_WithoutBF=np.hstack((Pre_x_train.reshape(-1,10240),train_label.reshape(-1,1)))
    trainSet_WithoutBF=np.hstack((trainSet_WithoutBF,Pre_y_train.reshape(-1,1)))
    # Split the Dynamic and Static samples (Without Bounding Features)#    
    x_trainDyStSplit_WithoutBF= np.array ( [trainSet_WithoutBF[train_Prelabel==i,:10240] for i in range(2)] )
    y_trainDyStSplit_WithoutBF= np.array ( [trainSet_WithoutBF[train_Prelabel==i,10240] for i in range(2)] )

## Training set preparation ###       
    testSet=np.hstack((x_test,y_test.reshape(-1,1)))
    testSet=np.hstack((testSet,Pre_y_test.reshape(-1,1)))
    
    testSet_WithoutBF=np.hstack((Pre_x_test.reshape(-1,10240),y_test.reshape(-1,1)))
    testSet_WithoutBF=np.hstack((testSet_WithoutBF,Pre_y_test.reshape(-1,1)))

    # Split the Dynamic and Static samples for testing (With Bounding Features)#       
    testDyorSt=PreClapredict
    x_testDyStSplit_WithBF= np.array ( [testSet[testDyorSt==i,:10260] for i in range(2)] )
    y_testDyStSplit_WithBF= np.array ( [testSet[testDyorSt==i,10260] for i in range(2)] )

    # Split the Dynamic and Static samples for testing (Without Bounding Features)#       
    x_testDyStSplit_WithoutBF= np.array ( [testSet_WithoutBF[testDyorSt==i,:10240] for i in range(2)] )
    y_testDyStSplit_WithoutBF= np.array ( [testSet_WithoutBF[testDyorSt==i,10240] for i in range(2)] )

#### static gesture classification ####
## static gesture dataset separation ##
    x_trainSt=x_trainDyStSplit_WithBF[0]
    y_trainSt=y_trainDyStSplit_WithBF[0]
    x_testSt=x_testDyStSplit_WithBF[0]
    y_testSt=y_testDyStSplit_WithBF[0]
       
## Match the wrongly preclassifed label with the true label for static gesture ##
    for i in range(len(testDyorSt)):
        if testSet[i,10261]==0 and testDyorSt[i]==1:
            y_testSt[i]=testSet[i,10260]
            
    x_St=np.vstack((x_trainSt,x_testSt))
    x_St=x_St.reshape(-1,1026)

## PCA: Reduce data dimension ##
    pcaModel=PCA(n_components=50)
    x_StPCA=pcaModel.fit_transform(x_St)
    x_trainStVol=x_trainSt.shape[0]*10 
    x_trainSt=x_StPCA[:x_trainStVol,:]
    x_testSt=x_StPCA[x_trainStVol:,:]

## Split frame set frame by frame for independent classification ##  
    y_St=np.hstack((y_trainSt,y_testSt))
    y_St_rep=[]
    for ele in y_St:
        for j in range(10):
            y_St_rep.append(ele)
    y_St_rep=np.array(y_St_rep)
    y_trainStVol=y_trainSt.shape[0]*10
    y_trainSt=y_St_rep[:y_trainStVol]

## Static gesture classifier training
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_trainSt,y_trainSt)
    
## Static gesture classifier prediction by voting to get a better result ##
    pred_st=[]
    for sample in range(int(x_testSt.shape[0]/10)):
        voteVector=list(knn.predict(x_testSt[sample*10:sample*10+9,:]))
        pred_st.append(max(voteVector, key=voteVector.count) )
    pred_st=np.array(pred_st)
    
#### Dynamic gesture classification ####
## Dynamic gesture data separation ##
    x_trainDy=x_trainDyStSplit_WithoutBF[1]
    y_trainDy=y_trainDyStSplit_WithoutBF[1]
    trainSet_Dy=np.hstack((x_trainDy,y_trainDy.reshape(-1,1)))
    x_trainJZSplit_WithoutBF= np.array ( [trainSet_Dy[y_trainDy==i,:10240] for i in range(10,12)] )
    
    y_testDy=y_testDyStSplit_WithoutBF[1]
    x_testDy=x_testDyStSplit_WithoutBF[1]
## Match the wrongly preclassifed label with the true label for dynamic gesture 
    for i in range(len(testDyorSt)):
        if testSet[i,10261]==1 and testDyorSt[i]==0:
            y_testDy[i-int(len(Pre_y_test)-sum(Pre_y_test))]=testSet[i,10260]

## Dynamic gesture classifier training and prediction ##    
    n_states_J=1
    n_states_Z=1
    HMM_J = hmm.GaussianHMM(n_components=n_states_J)
    HMM_Z = hmm.GaussianHMM(n_components=n_states_Z)   

    HMM_J.fit(x_trainJZSplit_WithoutBF[0])
    HMM_Z.fit(x_trainJZSplit_WithoutBF[1])

    score=[]
    for i in range(len(y_testDy)):
        score.append([(HMM_J.score(x_testDy[i,:].reshape(1,-1))),
                      (HMM_Z.score(x_testDy[i,:].reshape(1,-1)))])
    
    pred_dy=np.argmax(score,axis=1)+10
    
    pred=np.hstack((pred_st,pred_dy))

## re-insert the labels for these which are wrongly pre-classified ## 
    for i in range(len(y_testDy)):
        if y_testDy[i]>1:
            y_testDy[i]=y_testDy[i]-10

    y_testset=np.hstack((y_testSt,y_testDy+10))

## classification metrics evaluation ##   
    acc.append(accuracy_score(y_testset, pred))
    prec.append(precision_score(y_testset,pred,average='macro'))
    f1score.append(f1_score(y_testset,pred,average='macro'))
    recall.append(recall_score(y_testset,pred,average='macro'))    

### model save ####

## save pre-classifier ##
model.save('preclassifier_lstm.h5')

## save static gesture classifier and PCA model ##
joblib.dump(knn, 'static_classifier_knn.pkl')
joblib.dump(pcaModel, 'PCA.m')

## save dynamic gesture classifier ##
joblib.dump(HMM_J,'dynamic_classifier_hmm_J.pkl')
joblib.dump(HMM_Z, 'dynamic_classifier_hmm_Z.pkl')
   
# calculate macro average for metrics of pre-classification ##
PreaveAcc=sum(Preacc)/len(Preacc)
PreavePrec=sum(Preprec)/len(Preprec)
Preavef1score=sum(Pref1score)/len(Pref1score)
PreaveRecall=sum(Prerecall)/len(Prerecall)
print("Pre-Classifer CrossVal Acc: %.4f" %(PreaveAcc))
print("Pre-Classifer CrossVal Precision: %.4f" %(PreavePrec))
print("Pre-Classifer CrossVal f1-score: %.4f" %(Preavef1score))
print("Pre-Classifer CrossVal recall: %.4f" %(PreaveRecall))

## Draw confusion matrix for pre-classification ##
St_or_Dy={0:'Static',1:'Dynamic'}
confMat=confusion_matrix(Pre_y_test, PreClapredict)
confMat=pd.DataFrame(confMat)
confMat=confMat.rename(index=St_or_Dy,columns=St_or_Dy)
plt.figure(num='Confusion Matrix', facecolor='lightgray')
plt.title('Confusion Matrix (Pre-Classifier)', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()

## calculate macro average for metrics of classification ##
aveAcc=sum(acc)/len(acc)
avePrec=sum(prec)/len(prec)
avef1score=sum(f1score)/len(f1score)
aveRecall=sum(recall)/len(recall)
print("Hybird-12 CrossVal Acc: %.4f" %(aveAcc))
print("Hybird-12 CrossVal Precision: %.4f" %(avePrec))
print("Hybird-12 CrossVal f1-score: %.4f" %(avef1score))
print("Hybird-12 CrossVal recall: %.4f" %(aveRecall))

## Draw confusion matrix for classification ##
alphaDic_12={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'L',9:'U',10:'J',11:'Z'}
confMat=confusion_matrix(y_test, pred)
confMat=pd.DataFrame(confMat)
confMat=confMat.rename(index=alphaDic_12,columns=alphaDic_12)
plt.figure(num='Confusion Matrix', facecolor='lightgray')
plt.title('Confusion Matrix (proposed system)', fontsize=20)
ax=sns.heatmap(confMat, fmt='d',cmap='Greys',annot=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
plt.show()