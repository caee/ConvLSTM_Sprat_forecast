# -*- coding: utf-8 -*-
"""
Model library.

All models saved as classes to be called in main script. Also training loop

@author: Carl Emil Elling
08-07-2020
"""
import tensorflow as tf
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional,Conv3D, ConvLSTM2D,Dense,Dropout,Flatten, TimeDistributed, MaxPool2D,UpSampling2D,Concatenate,Conv2D,BatchNormalization


def fn_run_model(model,X_train,Y_train,xval,yval,batch_size,nb_epochs,model_name,working_path):
    #Function for fitting model to training and validation data
    history= tf.keras.callbacks.History()
    
    save_path= os.path.join(working_path,"model_weights/"+model_name+"/1/model.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            #Callback for saving weights 
            filepath=save_path, 
            verbose=1,
            monitor="val_loss", #Monitoring validation loss and saving only the best weights in a run
            save_weights_only=True,
            save_freq="epoch",
            save_best_only=True,
            mode="min")

    #Fitting the model
    history=model.fit(
            X_train,
            Y_train,
            batch_size=10,
            epochs=nb_epochs,
            verbose=1,
            callbacks=[cp_callback],
            validation_data=(xval,yval)
            )
    fig, ax1 = plt.subplots(1,1)
    ax1.set_title(model_name)
    ax1.plot(history.history["val_loss"],label='val_loss')
    ax1.plot(history.history["loss"],label='*loss*')
    ax1.legend()
    
#%% Custom Loss functions

def custom_rmse(true,pred):
    #given a target Y and prediction Y_hat, compute RMSE.
    #RMSE weighs large errors much higher than small errors, compared to MAE
    error=tf.sqrt(tf.reduce_mean(tf.square(true - pred)))
    return error

def weightedRMSELoss(w):
    #Weighs true zero-predictions lower than true nonzero predictions
    #From https://stackoverflow.com/questions/57618482/higher-loss-penalty-for-true-non-zero-predictions
    #Weights between 0 and 1. Lower weight means lower weight for true zero-predictions - essentially making the network train less on these 
    #weight above 1 means penalizing true nonzero predictions - that is making extra certain predictions might be true. Might lead to only zero-predictions.
    def loss(true, pred):
        
        #error has to be done in multiple steps, since tf.reduce_mean otherwise divides by zero when NN model is compiled without data
        error = tf.square(true - pred)
        error = tf.keras.backend.switch(tf.math.equal(true, 0), w**2 * error , error) #penalize error if zero. weight**2 to compensate for sqrt applied later
        error = tf.sqrt(tf.reduce_mean(error))
        return error

    return loss


"""
Loss function: Perceptual distance from 
https://www.youtube.com/watch?v=fmga0i0MXuU
"""
#def perceptual_distance(true, pred):
#    true *= 255.
#    pred *= 255.
#    rmean = (true[:, :, :, 0] + pred[:, :, :, 0]) / 2
#    r = true[:, :, :, 0] - pred[:, :, :, 0]
#    g = true[:, :, :, 1] - pred[:, :, :, 1]
#    b = true[:, :, :, 2] - pred[:, :, :, 2]
#
#    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))      
#%%
"""
Model 1
Low complexity architechture. Less deep network might help with smaller sample size
https://stackoverflow.com/questions/49540320/cuda-out-of-memory-when-training-convlstmd2d-model
"""

class model_1:
    def __init__(self,input_shape,output_shape=None,loss_type="mean_squared_error",optimizer="Adam",lr=0.003,kernel=4,noise=0.4):
        #creates model. 
        #kernel size essentially means capturing speed. To capture fast motions, a large kernel size is needed. [Shi et al 2015]
        self.model=tf.keras.Sequential([
                tf.keras.Input(
                        #shape=(None, 150, 174,1)
                        shape=input_shape #small data sample for testing purposes
                        ),# Variable-length sequence of 1x150x174 frames
                tf.keras.layers.GaussianNoise(noise),
                ConvLSTM2D(
                        filters=16, kernel_size=(kernel,kernel),
                        padding='same', return_sequences=True, 
                        activation='tanh', recurrent_activation='tanh',
                        kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                        data_format='channels_last', #Specifies which dimension of input data are the channels
                        dropout=0.6, recurrent_dropout=0.5, go_backwards=True,
                        stateful=False, #If true, the first frame in next batch will be last frame from previous
                        strides=(1, 1)
                        ),
                BatchNormalization(),
                

                ConvLSTM2D(
                        filters=1, kernel_size=(kernel,kernel),
                        padding='same', return_sequences=False, #Return sequences attribute tells whether output is a sequence, like the input, or a single frame
                        activation='sigmoid', recurrent_activation='tanh',
                        kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                        data_format='channels_last',
                        dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                        stateful=False, 
                        strides=(1, 1)
                        )   
                ])
        if optimizer=="Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr)    
        elif optimizer=="Adadelta":
            opt = "Adadelta"
        self.model.compile(loss=loss_type, optimizer=opt,metrics=['mse','mae'])
        self.model.summary()
    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="Model1"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
        #saving weights
        #name_weights = "final_model_fold" + str(j) + "_weights.h5"
    def predict(self,x):
        y=self.model.predict(x)
        return y
    def save_weights(self,path):
        self.model.save_weights(path)
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
        
        
#%%
"""
Model 2
Bi-directional ConvLSTMs. More exposure to data might help train the network
https://datascience.stackexchange.com/questions/27628/sliding-window-leads-to-overfitting-in-lstm
"""

class model_2:
    def __init__(self,input_shape,output_shape=None,loss_type="mean_squared_error",kernel=4):
        #creates model. 
        #kernel size essentially means capturing speed. To capture fast motions, a large kernel size is needed. [Shi et al 2015]
        self.model=tf.keras.Sequential([
                tf.keras.Input(
                        #shape=(None, 150, 174,1)
                        shape=input_shape #small data sample for testing purposes
                        ),# Variable-length sequence of 1x150x174 frames
                tf.keras.layers.GaussianNoise(0.4),        
                Bidirectional(ConvLSTM2D(
                        filters=8, kernel_size=(kernel,kernel),
                        padding='same', return_sequences=True, 
                        activation='tanh', recurrent_activation='tanh',
                        kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                        data_format='channels_last', #Specifies which dimension of input data are the channels
                        dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                        stateful=False, #If true, the first frame in next batch will be last frame from previous
                        strides=(1, 1)
                        )),
                BatchNormalization(),
                ConvLSTM2D(
                        filters=16, kernel_size=(kernel,kernel),
                        padding='same', return_sequences=True, 
                        activation='tanh', recurrent_activation='tanh',
                        kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                        data_format='channels_last', #Specifies which dimension of input data are the channels
                        dropout=0.2, recurrent_dropout=0.2, go_backwards=True,
                        stateful=False, #If true, the first frame in next batch will be last frame from previous
                        strides=(1, 1)
                        ),
                BatchNormalization(),

                ConvLSTM2D(
                        filters=1, kernel_size=(kernel,kernel),
                        padding='same', return_sequences=False, #Return sequences attribute tells whether output is a sequence, like the input, or a single frame
                        activation='sigmoid', recurrent_activation='tanh',
                        kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                        data_format='channels_last',
                        dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                        stateful=False, 
                        strides=(1, 1)
                        )   
                ])
        opt_adam = tf.keras.optimizers.Adam(learning_rate=0.005)    
        self.model.compile(loss=loss_type, optimizer=opt_adam,metrics=['mse','mae'])
        self.model.summary()
    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="Model2"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
        #saving weights
        #name_weights = "final_model_fold" + str(j) + "_weights.h5"
    def predict(self,x):
        y=self.model.predict(x)
        return y
    def save_weights(self,path):
        self.model.save_weights(path)
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation
        
#%%
"""
MODEL3:
    
Using deeper conv-LSTM network 

Simplified version of:
https://www.youtube.com/watch?v=MjFpgyWH-pk

"""
class model_3:
    def __init__(self,input_shape,output_shape=None,loss_type="mean_squared_error",kernel_size=3):
        self.n_filters=16
        
        self.model = tf.keras.Sequential([
        tf.keras.Input(input_shape,name='Input'),
        ConvLSTM2D(filters=16, kernel_size=(kernel_size,kernel_size),
                  padding='same', return_sequences=True, 
                  activation='tanh', recurrent_activation='tanh',
                  kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                  data_format='channels_last', #Specifies which dimension of input data are the channels
                  dropout=0.5, recurrent_dropout=0.5, go_backwards=True,
                  stateful=False, #If true, the first frame in next batch will be last frame from previous
                  strides=(1, 1)
                  ),
        ConvLSTM2D(filters=self.n_filters,kernel_size=(kernel_size,kernel_size),
                  padding='same',return_sequences=True,
                  activation='tanh', recurrent_activation='tanh',
                  kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                  data_format='channels_last', #Specifies which dimension of input data are the channels
                  dropout=0.5, recurrent_dropout=0.5, go_backwards=True,
                  stateful=False, #If true, the first frame in next batch will be last frame from previous
                  strides=(1, 1)
                  ),
#        ConvLSTM2D(filters=self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True),
        
        TimeDistributed(MaxPool2D(pool_size=(2,2))),
        
        ConvLSTM2D(filters=self.n_filters,kernel_size=(kernel_size,kernel_size),
                  padding='same',return_sequences=True,
                  activation='tanh', recurrent_activation='tanh',
                  kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                  data_format='channels_last', #Specifies which dimension of input data are the channels
                  dropout=0.5, recurrent_dropout=0.5, go_backwards=True,
                  stateful=False, #If true, the first frame in next batch will be last frame from previous
                  strides=(1, 1)
                  ),
        ConvLSTM2D(filters=self.n_filters,kernel_size=(kernel_size,kernel_size),
                  padding='same',return_sequences=True,
                  activation='tanh', recurrent_activation='tanh',
                  kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                  data_format='channels_last', #Specifies which dimension of input data are the channels
                  dropout=0.5, recurrent_dropout=0.5, go_backwards=True,
                  stateful=False, #If true, the first frame in next batch will be last frame from previous
                  strides=(1, 1)
                  ),
#        ConvLSTM2D(filters=self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True),
        
        TimeDistributed(UpSampling2D((2,2))),
        
        
        Conv3D(
            filters=1, kernel_size=(kernel_size,kernel_size,kernel_size),
            activation="sigmoid", padding="same"
        ),
        ConvLSTM2D(
            filters=1, kernel_size=(kernel_size, kernel_size),
            padding='same', return_sequences=False, #Return sequences attribute tells whether output is a sequence, like the input, or a single frame
            activation='sigmoid', recurrent_activation='tanh',
            kernel_initializer='glorot_uniform', unit_forget_bias=True, 
            dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
            stateful=False, 
            strides=(1, 1)
        )
        
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.005) 
        self.model.compile(loss=loss_type,optimizer=opt,metrics= ['mse','mae'])
        self.model.summary()
    def predict(self,x):
        x = self.model.predict(x)
        
        return x

    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="model3"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
    def summary(self):
        self.model.summary()
    def save_weights(self,path):
        self.model.save_weights(path)
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
#%%
"""
MODEL 4:
From Keras documentation
https://keras.io/examples/vision/conv_lstm/
https://medium.com/@rajin250/precipitation-prediction-using-convlstm-deep-neural-network-b9e9b617b436

Maybe add gaussian noise for robustness?

"""
class model_4():
    def __init__(self,input_shape,output_shape=None,loss_type='mean_squared_error',kernel_size=3):
        self.model = Sequential(
            [
                tf.keras.Input(
                    shape=input_shape #Input shape for network needs to have same dimensionality as output - except for batch size
                ),  
                ConvLSTM2D(
                    filters=8, kernel_size=(kernel_size, kernel_size),
                    padding='same', return_sequences=True, 
                    activation='tanh', recurrent_activation='hard_sigmoid',
                    kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                    dropout=0.2, recurrent_dropout=0.5, go_backwards=True,
                    stateful=False, #If true, the first frame in next batch will be last frame from previous
                    strides=(1, 1)
                ),
                BatchNormalization(),
                TimeDistributed(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")),
                ConvLSTM2D(
                    filters=10, kernel_size=(kernel_size, kernel_size),
                    padding='same', return_sequences=True, 
                    activation='tanh', recurrent_activation='tanh',
                    kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                    dropout=0.2, recurrent_dropout=0.5, go_backwards=True,
                    stateful=False, 
                    strides=(1, 1)
                ),
                BatchNormalization(),
                TimeDistributed(UpSampling2D(size=(2,2))),
                ConvLSTM2D(
                    filters=10, kernel_size=(kernel_size, kernel_size),
                    padding='same', return_sequences=True, #Return sequences attribute tells whether output is a sequence, like the input, or a single frame
                    activation='tanh', recurrent_activation='tanh',
                    kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                    dropout=0.2, recurrent_dropout=0.5, go_backwards=True,
                    stateful=False, 
                    strides=(1, 1)
                ),
                BatchNormalization(),
        
                Conv3D(
                    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
                ),
                                
                ConvLSTM2D(
                    filters=1, kernel_size=(kernel_size, kernel_size),
                    padding='same', return_sequences=False, #Return sequences attribute tells whether output is a sequence, like the input, or a single frame
                    activation='sigmoid', recurrent_activation='tanh',
                    kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                    dropout=0.1, recurrent_dropout=0.1, go_backwards=True,
                    stateful=False, 
                    strides=(1, 1)
                )
            ])

        
        opt = tf.keras.optimizers.Adam(learning_rate=0.005) 
        self.model.compile(optimizer=opt,loss=loss_type,metrics=['mse','mae'])
        self.model.summary()
    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="Model4"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
    def predict(self,x):
        y=self.model.predict(x)
        return y
    def summary(self):
        self.model.summary()
    def save_weights(self,path):
        self.model.save_weights(path)
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation
#TODO: add the prediction as ground truth to get predictions further in the future?

#%%
"""
MODEL 5:
Network inspired by:
https://www.youtube.com/watch?v=MjFpgyWH-pk
Input dimensions need to be even for this architechture to work
"""
class model_5():
    def __init__(self,input_shape,output_shape=None,loss_type='mean_squared_error',kernel_size=3):
        self.n_filters=8
        self.output_filters = 3
        self.n_downsamples = 1
        if input_shape[2]%2**self.n_downsamples!=0 or input_shape[3]%2**self.n_downsamples!=0:
            #Test whether input shape is still an int after downsampling with n (2x2) maxpoolings
            print("Input shape is no longer even after downsampling. Upsampling to same shape will not work")
            #If it is the case, zero pad the input shape?
            #First, finding the incompatible input dimension
#            dim1 = input_shape[2]%2**n_downsamples
#            dim2 = input_shape[3]%2**n_downsamples
    #        if dim1!=0:
    #            c1 = tf.keras.layers.ZeroPadding3D(padding=((dim1,0),(0,0),(0,0)))(c1)
    #        elif dim2!=0:
    #            c1 = tf.keras.layers.ZeroPadding3D(padding=((0,0),(dim2,0),(0,0)))(c1)
    #        else: 
    #            #if both are incompatible
    #            input_img = tf.keras.layers.ZeroPadding3D(padding=((dim1,0),(dim2,0),(0,0)))(input_img) 
            
        self.input_img = tf.keras.Input(input_shape,name='Input')
        
            #or reshape entire 
        self.x = ConvLSTM2D(filters=self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.input_img)
        self.x = ConvLSTM2D(filters=self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.x)
        self.c1 = ConvLSTM2D(filters=self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.x)

        self.x = TimeDistributed(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))(self.c1)
          
        self.x = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.x)
        self.x = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.x)
        self.c2 = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(3,3),padding='same',return_sequences=True)(self.x)
        
        self.x = TimeDistributed(UpSampling2D(size=(2,2)))(self.c2)
        self.x = tf.keras.layers.Concatenate()([self.c1,self.x])

        self.output = ConvLSTM2D(filters=1,kernel_size=(3,3),padding='same',return_sequences=False)(self.x)
        self.model = tf.keras.Model(self.input_img,self.output)
        
        opt_adam = tf.keras.optimizers.Adam(learning_rate=0.005) 
        self.model.compile(optimizer=opt_adam,loss=loss_type,metrics=['mse','mae'])
        self.model.summary()
    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="Model5"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
        #saving weights
        #name_weights = "final_model_fold" + str(j) + "_weights.h5"
    def predict(self,x):
        y=self.model.predict(x)
        return y
    def summary(self):
        self.model.summary()
    def save_weights(self,path):
        self.model.save_weights(path)
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation
        
        #%% Model 6
"""
MODEL 6:
Network inspired by model 5 with noise and conv3D
Input dimensions need to be even for this architechture to work. for is_int(input_dim/2**n), the higher n can be for this to be an int, the more feature-poolings can be done
"""
class model_6():
    def __init__(self,input_shape,output_shape=None,loss_type='mean_squared_error',kernel_size=3):
        self.n_filters=10
        self.output_filters=3
        self.n_downsamples = 1
        self.kernel_size = 3
        if input_shape[2]%2**self.n_downsamples!=0 or input_shape[3]%2**self.n_downsamples!=0:
            #Test whether input shape is still an int after downsampling with n (2x2) maxpoolings
            print("Input shape is no longer even after downsampling. Upsampling to same shape will not work")

        self.input_img = tf.keras.Input(input_shape,name='Input')
        self.x=tf.keras.layers.GaussianNoise(0.2)(self.input_img)
            #or reshape entire 
        self.x = ConvLSTM2D(filters=self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)
        self.x = ConvLSTM2D(filters=self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)
        self.c1 = ConvLSTM2D(filters=self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)

        self.x = TimeDistributed(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))(self.c1)
          
        self.x = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)
        self.x = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)
        self.c2 = ConvLSTM2D(filters=2*self.n_filters,kernel_size=(self.kernel_size,self.kernel_size),
                            padding='same',return_sequences=True,
                            data_format='channels_last',
                            recurrent_dropout=0.3)(self.x)
        
        self.x = TimeDistributed(UpSampling2D(size=(2,2)))(self.c2)
        self.x = tf.keras.layers.Concatenate()([self.c1,self.x])
        
        self.x = Conv3D(filters=self.output_filters,kernel_size=(self.kernel_size,self.kernel_size,self.kernel_size),
                        padding="same",data_format='channels_last')(self.x)
        
        self.output = ConvLSTM2D(filters=1,kernel_size=(self.kernel_size,self.kernel_size),
                                padding='same',return_sequences=False,
                                data_format='channels_last')(self.x)
        self.model = tf.keras.Model(self.input_img,self.output)
        
        opt_adam = tf.keras.optimizers.Adam(learning_rate=0.005) 
        self.model.compile(optimizer=opt_adam,loss=loss_type,metrics=['mse','mae'])
        self.model.summary()
    def runmodel(self,X,Y,X_hat,Y_hat,batch_size,nb_epochs,working_path,model_name="Model6"):
        fn_run_model(self.model,X,Y,X_hat,Y_hat,batch_size,nb_epochs,model_name,working_path)
        self.model.summary()
        #saving weights
        #name_weights = "final_model_fold" + str(j) + "_weights.h5"
    def predict(self,x):
        y=self.model.predict(x)
        return y
    def summary(self):
        self.model.summary()
    def save_weights(self,path):
        self.model.save_weights(path)
    def load_weights(self,checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Weights loaded")
    def evaluate(self,X_eval,Y_eval):
        evaluation = self.model.evaluate(X_eval,Y_eval)
        return evaluation