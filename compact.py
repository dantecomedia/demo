#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:32:57 2019

@author: rosa-mystica
"""
#LOADING DEPENDENCIES
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sklearn


class compact:
    
    def __init__(self,models):
        self.models=models
    def model(self,nos_layers,model_type,input_dimi, output_d,hidden_dim, hid_act,out_act,initi,loss,metrics,optimizer, dropout):
        self.nos_layers=nos_layers  #NUMBER OF LAYERS
        self.mode_type=model_type   #SEQUENTIAL / FUNCTIONAL
        self.input_dimi=input_dimi  # INPUT DIMENSION
        self.hidden_dim=hidden_dim  #HIDDEN DIMENSION
        self.output_d=output_d      #OUTPUT DIMENSION
        self.hid_act=hid_act        #HIDDEN ACTIVATION FUNCTION
        self.out_act=out_act        #OUTPUT ACTIVATION FUNCTION
        self.initi=initi            #INITAILIZATION
        self.loss=loss              #LOSS FUNCTION
        self.metrics=metrics        #METRICS LIST
        self.dropout=dropout        #DROPOUT
        self.optimizer=optimizer    #OPTIMIZER 
            
        from keras.layers import Input,Dense,Dropout
        if (model_type=='Sequential'):
            from keras.models import Sequential
            models=Sequential()         #calling model type
        models.add(Dense(output_dim = hidden_dim, init = initi , activation = hid_act, input_dim = input_dimi)) #first hidden layer
        models.add(Dropout(p=dropout))
                                    # Adding the second hidden layer
        for i in range(1,nos_layers-1):
            models.add(Dense(output_dim =hidden_dim , init = initi, activation = hid_act))
            models.add(Dropout(p=dropout))
        models.add(Dense(output_dim = output_d, init = initi, activation = out_act))
        
    
    def detail(self):
        models.summary()
        
    def fit(X,Y, epochs, batch_size):
        self.X=X
        self.Y=Y
        self.epochs=epochs
        self.batch_size=batch_size
        
        
    
