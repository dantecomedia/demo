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
        #def layers()
    #----------for 2D convulational layer----------------------
    def Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None):  
        from keras.layers import Conv2D
        if model!=None:
            model.add(Conv2D(filters,kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            return model
        else:
            x=general.pop()
            x.add(Conv2D(filters,kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            general.append(x)
            
    #----------for 1D convulational layer----------------------
    def Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None):
        from keras.layers import Conv1D
        if model!=None:
            model.add(Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            return model
        else:
            x=general.pop()
            x.add(Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            general.append(x)
            
    #-----------for 3D convulation layer---------------------
    def Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None):
        from keras.layers import Conv3D
        if model!=None:
            model.add(Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            return model
        else:
            x=general.pop()
            x.add(Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
            general.append(x)
    #----------------1D MAXPOOLING LAYER --------------------------
    def MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last',model=None):
        from keras.layers import MaxPooling1D
        if model!=None:
            model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
            general.append(x)
            
    #----------------2D MAXPOOLING LAYER----------------------------
    def MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,model=None):
        from keras.layers import MaxPooling2D
        if model!=None:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
            general.append(x)
            
        
    #-------------------3D MAXPOOLING LAYER--------------------------
    def MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None,model=None):
        from keras.layers import MaxPooling3D
        if model!=None:
            model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None))
            general.append(x)
            
    
    #----------------FLATTEN LAYER-----------------------------------
    def Flatten(model=None,data_format=None):
        from keras.layers import Flatten
        if model!=None:
            model.add(Flatten(data_format=None))
            return model
        else:
            x=general.pop()
            x.add(Flatten(data_format=None))
            general.append(x)
            
       #--------------RNN LAYER---------------------------------------------
    def RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None, input_shape=None):
        from keras.layers import RNN
        if model!=None:
            model.add(RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, input_shape=None))
            return model
        else:
            x=general.pop()
            x.add(RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
            general.append(x)
    
    
    
    #--------------------FULLY CONNECTED RNN LAYER---------------------------
    def SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None,input_shape=None):
        from keras.layers import SimpleRNN
        if model!=None:
            model=SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,input_shape=None)
            return model
        else:
            x=general.pop()
            x.add(SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
            general.append(x)
    
    #-------------------------GRU layer------------------------------------
    def GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False,input_shape=None,model=None):
        from keras.layers import GRU
        if model!=None:
            model=GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False,input_shape=None)
            return model
        else:
            x=general.pop()
            x.add(GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False))
            general.append(x)
            
    #--------------------LSTM layer----------------------------------------------
    def LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None,input_shape=None):
        from keras.layers import LSTM
        if model!=None:
            model.add(LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
            return model
        else:
            x=general.pop()
            x.add(LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
            general.append(x)
    
    #---------------------Convulation LSTM2D---------------------------------------
    def ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0,model=None,input_shape=None):
        from keras.layers import ConvLSTM2D
        if model!=None:
            model=ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
            return model
        else:
            top=general.pop()
            top.add(ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0,input_shape=None))
            general.append(top)
    
    #------------------------ Convulation LSTM 2D CELL LAYER--------------------------------
    def ConvLSTM2DCell(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,model=None,input_shape=None):
        from keras.layers import ConvLSTM2DCell
        if model!=None:
            model.add(ConvLSTM2DCell(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,input_shape=None))
            return model
        else:
            top=general.pop()
            top.add(ConvLSTM2DCell(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,input_shape=None))
            general.append(top)
    
    #---------------------------SIMPLE RNN CELL LAYER----------------------------------------
    def SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,model=None, input_shape=None):
        from keras.layers import SimpleRNNCell
        if model!=None:
            model.add(keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,input_shape=None))
            return model
        else:
            top=genral.pop()
            top.add(SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,input_shape=None))
            general.append(top)
    
    #----------------------------GRU CELL------------------
    def GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False,model=None,input_shape=None):
        from keras.layers import GRUCell
        if model!=None:
            model.add(GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False,input_shape=None))
            return model
        else:
            top=general.pop()
            top.add(keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False,input_shape=None))
            general.append(top)
    
    #------------------------LSTM CELL LAYER----------------------
    def LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,model=None,input_shape=None):
        if model!=None:
            model.add(LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,input_shape=None))
            return model
        else:
            top=general.pop()
            top.add(LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,input_shape=None))
            general.append(top)
    
    
    #------------------------CUDA DNN LAYER-----------------------------
    def CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,model=None,input_shape=None):
        from keras.layers import CuDNNGRU
        if model!=None:
            model.add(keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,input_shape=None))
            return model
        else:
            top=general.pop()
            top.add(CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,input_shape=None))
            general.append(top)
            
    #-------------------------CUDA LSTM LAYER---------------------------
    def CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,model=None, input_shape=None):
        from keras.layers import CuDNNLSTM
        if model!=None:
            model.add(CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,input_shape=None))
            return model
        else:
            top=general.pop()
            top.add(keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False))
            general.append(top)
            
    
    
    #----------------COMPILE FEATURE------------------------------------
    
    def compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=None):
        from keras.models import Model
        if model!=None:
            model.compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=None)
            return model
        else:
            x=general.pop()
            x.compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=None)
            general.append(x)
            
    #------------------FIT------------------------------------------------ 
    def fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,model=None):
        from keras.models import Model
        if model!=None:
            model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
            return model
        else:
            top=general.pop()
            top.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
            general.append(top)
    
    def evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, model=None):
        from keras.model import Model
        if model!=None:
            model.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
            return model
        else:
            top=general.pop()
            top.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
            general.append(top)
    
    

       
    def summary(model=None):
        if model!=None:
            general[model].summary()
        else :
            k=0
            for i in general:
                print("MODEL ",k,end="\n")
                i.summary()
                k=k+1    

        
    

        
#---------------------------CNN STANDALONE COMPACT MODEL------------------------------
general_cnn_layer=[]
class cnn:
    def model(Conv_type,nos_layers,MaxPooling_type,filters,kernel_size, input_shape=None,strides=(1, 1), dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model_type='Sequential',pool_size=2, MP_strides=None, padding='valid',data_format=None):
        from keras.models import Sequential
        if nos_layers==1:
            if model_type=='Sequential':
                cnn_model=Sequential()
                if Conv_type=='1D':
                    cnn_model=compact.Conv1D(filters, kernel_size, input_shape, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                elif Conv_type=='2D':
                    cnn_model=compact.Conv2D(filters, kernel_size, input_shape,strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                elif Conv_type=='3D':
                    cnn_model=compact.Conv3D(filters, kernel_size, input_shape,strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                if MaxPooling_type=='1D':
                    cnn_model=compact.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last',model=cnn_model)
                elif MaxPooling_type=='2D':
                    cnn_model=compact.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,model=cnn_model)
                elif MaxPooling_type=='3D':
                    cnn_model=compact.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None,model=cnn_model)
                
                cnn_model=compact.Flatten(data_format=None,model=cnn_model)
                general_cnn_layer.append(cnn_model)
                
        else:
            if model_type=='Sequential':
                cnn_model=Sequential()
            for i in range(nos_layers):
                if Conv_type[i]=='1D':
                    cnn_model=compact.Conv1D(filters[i], kernel_size[i], input_shape,strides=1,padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                    input_shape=None
                elif Conv_type[i]=='2D':
                    cnn_model=compact.Conv2D(filters[i], kernel_size[i], input_shape,strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                    input_shape=None
                elif Conv_type[i]=='3D':
                    cnn_model=compact.Conv3D(filters[i], kernel_size[i], input_shape,strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=cnn_model)
                    input_shape=None
                if MaxPooling_type[i]=='1D':
                    cnn_model=compact.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last',model=cnn_model)
                elif MaxPooling_type[i]=='2D':
                    cnn_model=compact.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,model=cnn_model)
                elif MaxPooling_type[i]=='3D':
                    cnn_model=compact.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None,model=cnn_model)
                
                cnn_model=compact.Flatten(data_format=None,model=cnn_model)
                general_cnn_layer.append(cnn_model)
                
                
            
            
            
            
            
            
            
            
    def compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        up=general_cnn_layer.pop()
        general_cnn_layer.append(compact.compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=up))
        
    def fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1):
        up=general_cnn_layer.pop()
        general_cnn_layer.append(compact.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,model=up))
    
    def evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None):
        up=general_cnn_layer.pop()
        general_cnn_layer.append(compact.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, model=up))
        
    def summary(model=general_cnn_layer):
        model.summary()
        
    
general_rnn_layer=[]      
class RNN:
    def model(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None):
        from keras.models import Sequential
        if model!=None:
            model=compact.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=model)
        
            
        
       

                
            
                


                
                
                
  

                
                    
                
                
        
        
        
        
        
        
    
        


#--------COMPACT LAYER VISUALISATION----------------------------------------
def final_desgin():
    from keras.utils import plot_model
    
    plot_model(model, to_file='model.png')
    for i in general:
        print(i)
        
