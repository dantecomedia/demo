#LOADING DEPENDENCIES
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sklearn
general=[]
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
            models=Sequential()  
            if isinstance(hidden_dim,list):
                    models.add(Dense(units = hidden_dim[0], kernel_initializer = kernel_initializer , activation = hid_act, input_dim = input_dim))
                    models.add(Dropout(rate=dropout))   # Adding the second hidden layer
                    for i in range(1,len(hidden_dim)):     #Adding hidden layers
                        models.add(Dense(units =hidden_dim[i], kernel_initializer = kernel_initializer, activation = hid_act))
                        models.add(Dropout(rate=dropout))
            else:
                models.add(Dense(units = hidden_dim, kernel_initializer = kernel_initializer , activation = hid_act, input_dim = input_dim))
                models.add(Dropout(rate=dropout))   # Adding the second hidden layer
                for i in range(1,nos_layers-1):     #Adding hidden layers
                    models.add(Dense(units =hidden_dim , kernel_initializer = kernel_initializer, activation = hid_act))
                    models.add(Dropout(rate=dropout))
            models.add(Dense(units = output_dim, kernel_initializer = kernel_initializer, activation = out_act))
            models.compile(optimizer=optimizer,loss=loss,metrics=metrics)
            feed_forward.append(models)
    
    def detail(self):
        top=feed_forward.pop()
        top.summary()
        feed_forward.append(top)
    
    def fit(self,X,Y, epochs, batch_size):
        self.X=X
        self.Y=Y
        self.epochs=epochs
        self.batch_size=batch_size
        top=feed_forward.pop()
        top.fit(X,Y,batch_size=batch_size,epochs=epochs)
        feed_forward.append(top)
    def predict(self,X):
        self.X=X
        return feed_forward[0].predict(X)

#general=[]  #storing model in stack
class core():
    #global general
    from keras.layers import Dense
    from keras.models import Model

    def __init__(self):
     self.general=[]
    def Sequential(self,models=None): 
        self.models=models # FOR SEQUENTIAL MODEL
        from keras.models import Sequential
        if models==None:# CALLING THE SEQUENTIAL KERAS LIBRARY
            models=Sequential()       #Creating the instance of the sequential class
            general.append(models)
           # print("APPENDED")
            #print(general[0])
        return general#Appending the model to the stack
        #def layers()
    #-------------------Dense layer---------------------------
    def Dense(self,units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None):
        self.units=units
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint
        self.model=model
        from keras.layers import Dense
        if model!=None:
            model.add(Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            return model
        else:
            top=general.pop()
            top.add(Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            general.append(top)
        
        
        
    #----------for 2D convulational layer----------------------
    def Conv2D(self,filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,input_shape=None,model=None):  
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint
        self.input_shape=input_shape

        
        from keras.layers import Conv2D
        if model!=None:
            model.add(Conv2D(filters=filters,kernel_size=kernel_size, input_shape=input_shape,strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            return model
        else:
            x=general.pop()
            x.add(Conv2D(filters=filters,kernel_size=kernel_size, input_shape=input_shape, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            general.append(x)
            
    #----------for 1D convulational layer----------------------
    def Conv1D(self,filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint


        from keras.layers import Conv1D
        if model!=None:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            return model
        else:
            x=general.pop()
            x.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint))
            general.append(x)
            
    #-----------for 3D convulation layer---------------------
    def Conv3D(self,filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model=None,input_dim=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint
        self.model=model
        self.input_dim=input_dim




        from keras.layers import Conv3D
        if model!=None:
            model.add(Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,input_dim=input_dim))
            return model
        else:
            x=general.pop()
            x.add(Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,input_dim=input_dim))
            general.append(x)
    #----------------1D MAXPOOLING LAYER --------------------------
    def MaxPooling1D(self,pool_size=2, strides=None, padding='valid', data_format='channels_last',model=None):
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.model=model

        from keras.layers import MaxPooling1D
        if model!=None:
            model.add(MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            general.append(x)
            
    #----------------2D MAXPOOLING LAYER----------------------------
    def MaxPooling2D(self,pool_size=(2, 2), strides=None, padding='valid', data_format=None,model=None):
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.model=model

        from keras.layers import MaxPooling2D
        if model!=None:
            model.add(MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            general.append(x)
            
        
    #-------------------3D MAXPOOLING LAYER--------------------------
    def MaxPooling3D(self,pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None,model=None):
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.model=model

        from keras.layers import MaxPooling3D
        if model!=None:
            model.add(MaxPooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            return model
        else:
            x=general.pop()
            x.add(MaxPooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
            general.append(x)
            
    
    #----------------FLATTEN LAYER-----------------------------------
    def Flatten(self,model=None,data_format=None):
        self.model=model
        self.data_format=data_format
        from keras.layers import Flatten
        if model!=None:
            model.add(Flatten(data_format=data_format))
            return model
        else:
            x=general.pop()
            x.add(Flatten(data_format=data_format))
            general.append(x)
            
       #--------------RNN LAYER---------------------------------------------
    def RNN(self,cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None, input_shape=None):
        self.cell=cell
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.unroll=unroll
        self.model=model
        self.input_shape=input_shape

        from keras.layers import RNN
        if model!=None:
            model.add(RNN(cell=cell, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll, input_shape=input_shape))
            return model
        else:
            x=general.pop()
            x.add(RNN(cell=cell, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll,input_shape=input_shape))
            general.append(x)
    
    
    
    #--------------------FULLY CONNECTED RNN LAYER---------------------------
    def SimpleRNN(self,units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None,input_shape=None):
        self.units=units
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.unroll=unroll



        from keras.layers import SimpleRNN
        if model!=None:
            model=SimpleRNN(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll,input_shape=input_shape)
            return model
        else:
            x=general.pop()
            x.SimpleRNN(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll,input_shape=input_shape)
            general.append(x)
    
    #-------------------------GRU layer------------------------------------
    def GRU(self,units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False,input_shape=None,model=None):
        self.units=units
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.activity_regularize=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.implementation=implementation
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.unroll=unroll
        self.reset_after=reset_after



        from keras.layers import GRU
        if model!=None:
            model=GRU(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll, reset_after=reset_after,input_shape=input_shape)
            return model
        else:
            x=general.pop()
            x.GRU(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll, reset_after=reset_after,input_shape=input_shape)
            general.append(x)
            
    #--------------------LSTM layer----------------------------------------------
    def LSTM(self,units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None,input_shape=None):
        self.units=units
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.unit_forget_bias=unit_forget_bias
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.implementation=implementation
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.unroll=unroll
        self.input_shape=input_shape
        self.model=model


        from keras.layers import LSTM
        if model!=None:
            model.add(LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll,input_shape=input_shape))
            return model
        else:
            x=general.pop()
            x.add(LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll))
            general.append(x)
    
    #---------------------Convulation LSTM2D---------------------------------------
    def ConvLSTM2D(self,filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0,model=None,input_shape=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.unit_forget_bias=unit_forget_bias
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.return_sequences=return_sequences
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.model=model
        self.input_shape=input_shape




        from keras.layers import ConvLSTM2D
        if model!=None:
            model=ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=orthogonal, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, go_backwards=go_backwards, stateful=stateful, dropout=dropout, recurrent_dropout=recurrent_dropout,input_shape=input_shape)
            return model
        else:
            top=general.pop()
            top.ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=orthogonal, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, go_backwards=go_backwards, stateful=stateful, dropout=dropout, recurrent_dropout=recurrent_dropout)
            general.append(top)
    
    #------------------------ Convulation LSTM 2D CELL LAYER--------------------------------
    def ConvLSTM2DCell(self,filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,model=None,input_shape=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.unit_forget_bias=unit_forget_bias
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.return_sequences=return_sequences
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.model=model
        self.input_shape=input_shape


        from keras.layers import ConvLSTM2DCell
        if model!=None:
            model.add(ConvLSTM2DCell(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(ConvLSTM2DCell(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout))
            general.append(top)
    
    #---------------------------SIMPLE RNN CELL LAYER----------------------------------------
    def SimpleRNNCell(self,units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,model=None, input_shape=None):
        self.units=units
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.model=model
        self.input_shape=input_shape

        from keras.layers import SimpleRNNCell
        if model!=None:
            model.add(SimpleRNNCell(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(SimpleRNNCell(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout))
            general.append(top)
    
    #----------------------------GRU CELL------------------
    def GRUCell(self,units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False,model=None,input_shape=None):
        self.units=units
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_regularizer=bias_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.implementation=implementation
        self.reset_after=reset_after
        self.model=model
        self.input_shape=input_shape

        from keras.layers import GRUCell
        if model!=None:
            model.add(GRUCell(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, reset_after=reset_after,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(GRUCell(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation, reset_after=reset_after))
            general.append(top)
    
    #------------------------LSTM CELL LAYER----------------------
    def LSTMCell(self,units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,model=None,input_shape=None):
        self.units=units
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.unit_forget_bias=unit_forget_bias
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.implementation=implementation
        self.model=model
        self.input_shape=input_shape




        if model!=None:
            model.add(LSTMCell(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(LSTMCell(units=units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation))
            general.append(top)
    
    
    #------------------------CUDA DNN LAYER-----------------------------
    def CuDNNGRU(self,units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,model=None,input_shape=None):
        self.units=units
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.stateful=stateful
        self.model=model
        self.input_shape=input_shape

        from keras.layers import CuDNNGRU
        if model!=None:
            model.add(CuDNNGRU(units=units, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, stateful=stateful,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(CuDNNGRU(units=units, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, stateful=stateful))
            general.append(top)
            
    #-------------------------CUDA LSTM LAYER---------------------------
    def CuDNNLSTM(self,units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False,model=None, input_shape=None):
        self.units=units
        self.kernel_initializer=kernel_initializer
        self.recurrent_initializer=recurrent_initializer
        self.bias_initializer=bias_initializer
        self.unit_forget_bias=unit_forget_bias
        self.kernel_regularizer=kernel_regularizer
        self.recurrent_regularizer=recurrent_regularizer
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.kernel_constraint=kernel_constraint
        self.recurrent_constraint=recurrent_constraint
        self.bias_constraint=bias_constraint
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.stateful=stateful
        self.model=model
        self.input_shape=input_shape

        from keras.layers import CuDNNLSTM
        if model!=None:
            model.add(CuDNNLSTM(units=units, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, stateful=stateful,input_shape=input_shape))
            return model
        else:
            top=general.pop()
            top.add(CuDNNLSTM(units=units, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, stateful=stateful))
            general.append(top)
            
    #-----------------Dropout Layer------------------------------------
    def Dropout(self,rate):
        self.rate=rate
        from keras.layers import Dense
        from keras.layers import Dropout
        top=general.pop()
        top.add(Dropout(rate=rate))
        general.append(top)
                
    
    #----------------COMPILE FEATURE------------------------------------
    
    def compile(self,optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=None):
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.loss_weights=loss_weights
        self.sample_weight_mode=sample_weight_mode
        self.weighted_metrics=weighted_metrics
        self.target_tensors=target_tensors
        self.model=model



        if model!=None:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors)
            return model
        else:
            x=general.pop()
            x.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors)
            general.append(x)
            
    #------------------FIT------------------------------------------------ 
    def fit(self,x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,model=None):
        self.batch_size=batch_size
        self.epochs=epochs
        self.x=x
        self.y=y
        self.verbose=verbose
        self.callbacks=callbacks
        self.validation_split=validation_split
        self.validation_data=validation_data
        self.shuffle=shuffle
        self.class_weight=class_weight
        self.sample_weight=sample_weight
        self.initial_epoch=initial_epoch
        self.steps_per_epoch=steps_per_epoch
        self.validation_steps=validation_steps
        self.validation_freq=validation_freq
        self.model=model

        from keras.models import Model
        if model!=None:
            model=model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq)
            return model
        else:
            top=general.pop()
            top.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq)
            general.append(top)
    
    def evaluate(self,x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, model=None):
        self.x=x
        self.y=y
        self.batch_size=batch_size
        self.verbose=verbose
        self.sample_weight=sample_weight
        self.steps=steps
        self.callbacks=callbacks
        self.model=model


        from keras.model import Model
        if model!=None:
            model.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps, callbacks=callbacks)
            return model
        else:
            top=general.pop()
            top.evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps, callbacks=callbacks)
            general.append(top)
    
    

       
    def summary(self,model=None):
        self.model=model
        if model!=None:
            general[model].summary()
        else :
            top=general.pop()
            top.summary()
            general.append(top)

        
    
    

        
#---------------------------CNN STANDALONE core MODEL------------------------------
general_cnn_layer=[]
class cnn:
    def model(self,Conv_type,nos_layers,MaxPooling_type,filters,kernel_size, input_shape=None,strides=(1, 1), dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,model_type='Sequential',pool_size=2, MP_strides=None, padding='valid',data_format=None):
        self.Conv_type=Conv_type
        self.nos_layers=nos_layers
        self.MaxPooling_type=MaxPooling_type
        self.filters=filters
        self.kernel_size=kernel_size
        self.input_shape=input_shape
        self.strides=strides
        self.dilation_rate=dilation_rate
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.data_format=data_format
        self.padding=padding
        self.MP_strides=MP_strides
        self.pool_size=pool_size
        self.model_type=model_type
        self.bias_constraint=bias_constraint
        self.kernel_constraint=kernel_constraint
        self.activity_regularizer=activity_regularizer
        self.bias_regularizer=bias_regularizer
        self.bias_initializer=bias_initializer

        from keras.models import Sequential
        if nos_layers==1:
            if model_type=='Sequential':
                cnn_model=Sequential()
                if Conv_type=='1D':
                    cnn_model=core.Conv1D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, strides=1, padding=padding, data_format='channels_last', dilation_rate=1, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                elif Conv_type=='2D':
                    cnn_model=core.Conv2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,strides=(1, 1), padding=padding, data_format=data_format, dilation_rate=(1, 1), activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                elif Conv_type=='3D':
                    cnn_model=core.Conv3D(filters=filters, kernel_size=kernel_size, input_shape=input_shape,strides=(1, 1, 1), padding=padding, data_format=data_format, dilation_rate=(1, 1, 1), activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                if MaxPooling_type=='1D':
                    cnn_model=core.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last',model=cnn_model)
                elif MaxPooling_type=='2D':
                    cnn_model=core.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=data_format,model=cnn_model)
                elif MaxPooling_type=='3D':
                    cnn_model=core.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=data_format,model=cnn_model)
                
                cnn_model=core.Flatten(data_format=None,model=cnn_model)
                general_cnn_layer.append(cnn_model)
                
        else:
            if model_type=='Sequential':
                cnn_model=Sequential()
            for i in range(nos_layers):
                if Conv_type[i]=='1D':
                    cnn_model=core.Conv1D(filters=filters[i], kernel_size=kernel_size[i], input_shape=input_shape,strides=1,padding=padding, data_format='channels_last', dilation_rate=1, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                    input_shape=None
                elif Conv_type[i]=='2D':
                    cnn_model=core.Conv2D(filters=filters[i], kernel_size=kernel_size[i], input_shape=input_shape,strides=(1, 1), padding=padding, data_format=data_format, dilation_rate=(1, 1), activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                    input_shape=None
                elif Conv_type[i]=='3D':
                    cnn_model=core.Conv3D(filters=filters[i], kernel_size=kernel_size[i], input_shape=input_shape,strides=(1, 1, 1), padding=padding, data_format=data_format, dilation_rate=(1, 1, 1), activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,model=cnn_model)
                    input_shape=None
                if MaxPooling_type[i]=='1D':
                    cnn_model=core.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last',model=cnn_model)
                elif MaxPooling_type[i]=='2D':
                    cnn_model=core.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=data_format,model=cnn_model)
                elif MaxPooling_type[i]=='3D':
                    cnn_model=core.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=data_format,model=cnn_model)
                
                cnn_model=core.Flatten(data_format=None,model=cnn_model)
                general_cnn_layer.append(cnn_model)
                
                
            
            
            
            
            
            
            
            
    def compile(self,optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.loss_weights=loss_weights
        self.sample_weight_mode=sample_weight_mode
        self.weighted_metrics=weighted_metrics
        self.target_tensors=target_tensors



        up=general_cnn_layer.pop()
        general_cnn_layer.append(core.compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None,model=up))
        
    def fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1):
        up=general_cnn_layer.pop()
        general_cnn_layer.append(core.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,model=up))
    
    def evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None):
        up=general_cnn_layer.pop()
        general_cnn_layer.append(core.evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, model=up))
        
    def summary(model=general_cnn_layer):
        model.summary()
        
    
general_rnn_layer=[]      
class RNN:
    def model(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=None):
        self.cell=cell
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards=go_backwards
        self.stateful=stateful
        self.unroll=unroll
        self.model=model

        from keras.models import Sequential
        if model!=None:
            model=core.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=model)
            return model
        else:
            top=general_rnn_layer.pop()
            general_rnn_layer.append(core.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,model=top))
            return general_rnn_layer


        
            
#--------core LAYER VISUALISATION----------------------------------------
def final_desgin():
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    for i in general:
        print(i)
        
