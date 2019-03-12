#LOADING DEPENDENCIES 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import keras
import matplotlib.pyplot as plt 

class compact:
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
        from keras.models import model_type     
        from keras.layers import Input,Dense
        models=model_type()         #calling model type
        models.add(Dense(output_dim = hidden_dim, init = initi , activation = hid_act, input_dim = input_dimi)) #first hidden layer
        models.add(Dropout(p=dropout))
                                    # Adding the second hidden layer
        for i in range(1,nos_layer-1):
            models.add(Dense(output_dim =hidden_dim , init = initi, activation = hid_act))
            models.add(Droput(p=dropout))
        models.add(Dense(output_dim = output_d, init = initi, activation = out_act))


        
    def fit(X,Y, epochs, batch_size):
        self.X=X
        self.Y=Y
        self.epochs=epochs
        self.batch_size=batch_size
        
