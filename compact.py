#LOADING DEPENDENCIES
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sklearn

feed_forward=[]
class feedforward:
    def model(self,nos_layers,model_type,input_dim, output_dim,hidden_dim, hid_act,out_act,kernel_initializer,loss,metrics,optimizer, dropout):
        self.nos_layers=nos_layers  #NUMBER OF LAYERS
        self.mode_type=model_type   #SEQUENTIAL / FUNCTIONAL
        self.input_dim=input_dim  # INPUT DIMENSION
        self.hidden_dim=hidden_dim  #HIDDEN DIMENSION
        self.output_dim=output_dim      #OUTPUT DIMENSION
        self.hid_act=hid_act        #HIDDEN ACTIVATION FUNCTION
        self.out_act=out_act        #OUTPUT ACTIVATION FUNCTION
        self.kernel_initializer=kernel_initializer #INITAILIZATION
        self.loss=loss              #LOSS FUNCTION
        self.metrics=metrics        #METRICS LIST
        self.dropout=dropout        #DROPOUT
        self.optimizer=optimizer    #OPTIMIZER 
            
        from keras.layers import Input,Dense,Dropout
        if (model_type=='Sequential'):
            from keras.models import Sequential
            models=Sequential()         #calling model type
        models.add(Dense(units = hidden_dim, kernel_initializer = kernel_initializer , activation = hid_act, input_dim = input_dim))
        models.add(Dropout(rate=dropout))   # Adding the second hidden layer
        for i in range(1,nos_layers-1):     #Adding hidden layers
            models.add(Dense(units =hidden_dim , kernel_initializer = kernel_initializer, activation = hid_act))
            models.add(Dropout(rate=dropout))
        models.add(Dense(units = output_dim, kernel_initializer = kernel_initializer, activation = out_act))
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

 #iNTRODUCING ALL THE FEATURES OF THE LAYERS 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
general=[]  #storing model as stack
class compact:
    def Sequential():    # FOR SEQUENTIAL MODEL
        from keras.models import Sequential   # CALLING THE SEQUENTIAL KERAS LIBRARY
        models=Sequential()       #Creating the instance of the sequential class
        general.append(models)    #Appending the model to the stack
    #----------for 2D convulational layer----------------------
    def Conv2D(model=None,filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):  
        from keras.layers import Conv2D   
        x=general.pop()
        x.add(Conv2D(filters,kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        general.append(x)
    #----------for 1D convulational layer----------------------
    def Conv1D(model=None,filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        x=general.pop()
        from keras.layers import Conv1D
        x.add(Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        general.append(x)
    #-----------for 3D convulation layer---------------------
    def Conv3D(model=None,filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        x=general.pop()
        from keras.layers import Conv3D
        x.add(Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
        general.append(x)
    #----------------1D MAXPOOLING LAYER --------------------------
    def MaxPooling1D(model=None,pool_size=2, strides=None, padding='valid', data_format='channels_last'):
        from keras.layers import MaxPooling1D
        x=general.pop()
        x.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
        general.append(x)
    #----------------2D MAXPOOLING LAYER----------------------------
    def MaxPooling2D(model=None,pool_size=(2, 2), strides=None, padding='valid', data_format=None):
        from keras.layers import MaxPooling2D
        x=general.pop()
        x.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        general.append(x)
    
    #-------------------3D MAXPOOLING LAYER--------------------------
    def MaxPooling3D(model=None,pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None):
        from keras.layers import MaxPooling3D
        x=general.pop()
        x.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None))
        general.append(x)
    
    #----------------FLATTEN LAYER-----------------------------------
    def Flatten(model=None,data_format=None):
        from keras.layers import Flatten
        x=general.pop()
        x.add(Flatten(data_format=None))
        general.append(x)
   
    def detail(model=None):
        if model!=None:
            general[model].summary()
        else :
            k=0
            for i in general:
                print("MODEL ",k,end="\n")
                i.summary()
                k=k+1
        
    
    
        
        
#---------------------------CNN STANDALONE COMPACT MODEL------------------------------
class cnn:
    def model(model_type,*Conv ):
        from keras.models import Sequential
        if model_type='Sequential':
            cnn_model=Sequential()
            for i in Conv:
                if i 
                
        
        
        
        
        
        
    
        


#--------COMPACT LAYER VISUALISATION----------------------------------------
def final_desgin():
    from keras.utils import plot_model
    
    plot_model(model, to_file='model.png')
    for i in general:
        print(i)
        
