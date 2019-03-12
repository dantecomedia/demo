#LOADING DEPENDENCIES
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sklearn

feed_forward=[]
class compact:
    
    
    def model(self,nos_layers,model_type,input_dim, output_dim,hidden_dim, hid_act,out_act,init,loss,metrics,optimizer, dropout):
        self.nos_layers=nos_layers  #NUMBER OF LAYERS
        self.mode_type=model_type   #SEQUENTIAL / FUNCTIONAL
        self.input_dim=input_dim  # INPUT DIMENSION
        self.hidden_dim=hidden_dim  #HIDDEN DIMENSION
        self.output_dim=output_dim      #OUTPUT DIMENSION
        self.hid_act=hid_act        #HIDDEN ACTIVATION FUNCTION
        self.out_act=out_act        #OUTPUT ACTIVATION FUNCTION
        self.init=init            #INITAILIZATION
        self.loss=loss              #LOSS FUNCTION
        self.metrics=metrics        #METRICS LIST
        self.dropout=dropout        #DROPOUT
        self.optimizer=optimizer    #OPTIMIZER 
            
        from keras.layers import Input,Dense,Dropout
        if (model_type=='Sequential'):
            from keras.models import Sequential
            models=Sequential()         #calling model type
        models.add(Dense(output_dim = hidden_dim, init = init , activation = hid_act, input_dim = input_dim)) #first hidden layer
        models.add(Dropout(p=dropout))
                                    # Adding the second hidden layer
        for i in range(1,nos_layers-1):
            models.add(Dense(output_dim =hidden_dim , init = init, activation = hid_act))
            models.add(Dropout(p=dropout))
        models.add(Dense(output_dim = output_dim, init = init, activation = out_act))
        models.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        feed_forward.append(models)
    
    def detail(self):
        feed_forward[0].summary()
        
    def fit(self,X,Y, epochs, batch_size):
        self.X=X
        self.Y=Y
        self.epochs=epochs
        self.batch_size=batch_size
        feed_forward[0].fit(X,Y,batch_size=batch_size,epochs=epochs)
    def predict(self,X):
        self.X=X
        return feed_forward[0].predict(X)
        
        
    
