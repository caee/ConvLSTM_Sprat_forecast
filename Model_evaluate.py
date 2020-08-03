# -*- coding: utf-8 -*-
"""
Model Evaluation.

Library for evaluation functions.
@author: Carl Emil Elling
08-07-2020
"""
import numpy as np
import tensorflow as tf

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import Model_lib
#%% Load model:

def model_load(model,input_shape,checkpoint_path,loss_type='mean_squared_error'):
    model=Model_lib.model(input_shape,loss_type)
    model.load_weights(checkpoint_path)
    return model

#%%MODEL EVALUATION
"""
Evaluation of model. Based on Zero Normalized Cross Correlation. A perfect correlation is 1, a perfect anticorrelation is -1
seeing the correlation over time - to check for actual prediction or just learning prev step
"""
t=1
def eval_ZNCC(model,X,Y):
    #Evaluates a model based on zero-normalized cross correlation between target and prediction.
    #Also returns the ZNCC for prediction and target time-shifted 1 spot. 
    #This to check whether model accuracy is only a figment of it copying input
    totcor = []
    shiftcor = []
    for i in range(np.size(Y,axis=0)):
        #loop over all targets
        pred = model.predict(X[i:i+1,:,:,:,:])

    
        #zero-normalized cross correlation
        im1 = Y[i,:,:,0]
        im2 = pred[0,:,:,0]
        if np.std(im1)!=0 and np.std(im2)!=0:
            im1 = (im1-np.mean(im1))/np.std(im1)
            im2 = (im2-np.mean(im2))/np.std(im2)
            cor  = np.sum(im1*im2)/np.size(im1) #ZNCC
            totcor.append(cor) #appended to a list to be able to plot it
        # elif np.std(im1)=0 and np.std(im2)=0:
        #     #If both images are 0, total correlation
        #     totcor.append(1)
            #If one image is only zeroes, it doesn't make sense to correlate with another. Thus, this ZNCC only evaluates how well images that are NOT zero fit

        #Correlation with 1-shifted input
    for i in range(np.size(Y,axis=0)-t):
        pred = model.predict(X[i:i+1,:,:,:,:])
        im1 = Y[i-t,:,:,0] #shifting target 1 timestep forward to check whether model only learns to copy data
        im2 = pred[0,:,:,0]
        if np.std(im1)!=0 and np.std(im2)!=0:
            im1 = (im1-np.mean(im1))/np.std(im1)
            im2 = (im2-np.mean(im2))/np.std(im2)
            cor  = np.sum(im1*im2)/np.size(im1)
            shiftcor.append(cor)
        # elif np.std(im1)=0 and np.std(im2)=0:
        #     #If both images are 0, total correlation
        #     totcor.append(1)
        
    totcor_sum=sum(totcor)/len(totcor)
    shiftcor_sum=sum(shiftcor)/len(shiftcor)
    return totcor,shiftcor,totcor_sum,shiftcor_sum


#%% Baseline set
def createBaseline(test_set):
    #creates a baseline if the network only predicts 0-s. Deprecated, since cross correlation with zero-data is not possible 
    totcor = []
    baseline=np.zeros(np.shape(test_set)) #our 0-prediction network
    for i in range(np.size(test_set,axis=0)):
        #loop over all targets
        
        #zero-normalized cross correlation
        im1 = baseline[i,:,:,0]
        im2 = test_set[i,:,:,0]
        if np.std(im1)!=0 and np.std(im2)!=0:
            im1 = (im1-np.mean(im1))/np.std(im1)
            im2 = (im2-np.mean(im2))/np.std(im2)
            cor  = sum(im1*im2*1/np.size(im1)) #ZNCC
            totcor.append(cor) #appended to a list to be able to plot it
        else:
            totcor.append(1)
        return totcor

#%%
def evaluate(model,model_name,x_test,target_test):
    #Creates evaluation array
    #Inputs model, and test set (X,Y)
    
    #First, standard evaluation on loss (rather useless comparison since different loss functions) mse and mae
    evaluation=model.evaluate(x_test,target_test)
    #Then, ZNCC. Also differ ence between ZNCC and time shifted ZNCC. If positive, ZNCC is better predictor than time shifted ZNCC
    totcor,shiftcor,totcor_sum,shiftcor_sum=eval_ZNCC(model,x_test,target_test)
    evaluation=np.append(evaluation,totcor_sum) #ZNCC
    evaluation=np.append(evaluation,shiftcor_sum) #shifted ZNCC
    evaluation=np.append(evaluation,totcor_sum-shiftcor_sum) #ZNCC difference
    evaluation=np.append(evaluation,model_name) #appending model name
    return evaluation