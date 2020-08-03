# -*- coding: utf-8 -*-
"""
Inspiration from MIT Deep learning & https://www.youtube.com/watch?v=fmga0i0MXuU

@author: Carl Emil Elling
08-07-2020
Resized data from the start. Single frame as target. Multiple channels of data as input

Remember to set this script as working directory
"""

#Arrays
import numpy as np

#Neural network construction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional,Conv3D, ConvLSTM2D,Dense,Dropout,Flatten, TimeDistributed, MaxPool2D,UpSampling2D,Concatenate,Conv2D,BatchNormalization

#from tensorflow.keras.normalization import BatchNormalization
import netCDF4 as netcdf #For importing C-squares data
import cv2 #for resizing

#Figures
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import cartopy.crs as ccrs #map projections
import cartopy.feature as cfeature #features in maps
import seaborn as sns #Color for maps
from scipy import signal #For correlation'


from pathlib import Path


import Model_lib
import Model_evaluate


pause=False #For pausing animations

#%%
def normalized_netcdf_variable(ncf, varname, accept_range):
    """
    Normalizing variables in a range. Credit: Asbjørn Christensen @ DTU Aqua
    """
    # masking load by netCDF4 apparently fails; directly work with var.data and manually detect fills
    # scaled variable to range [0,1]; project fills to 0
    # project values outside accept_range to 0, 1 respectively
    var  = ncf.variables[varname]
    ymin = max(np.amin(np.where(var[:].data==var._FillValue,  1.e20, var[:].data)), accept_range[0]) # enforced min
    ymax = min(np.amax(np.where(var[:].data==var._FillValue, -1.e20, var[:].data)), accept_range[1]) # enforced max
    var  = np.where(var[:].data==var._FillValue, ymin, var[:].data) # project fills to lower limit
    scaled_var = (var-ymin)/(ymax-ymin) # rescale
    scaled_var = np.where(scaled_var<0, 0, scaled_var)
    scaled_var = np.where(scaled_var>1, 1, scaled_var)
    print("normalized_netcdf_variable: %f < %s < %f" % (ymin, varname, ymax))
    return scaled_var


def nint(x): return int(round(x))


# -----------  load coaxial signal+response data -----------
working_path=str(Path(__file__).parent)

data_name = "ns_input_02Oct2019_postedited.nc"
data_path = str(Path(working_path,data_name))

ncf_env = netcdf.Dataset(data_path, "r")
cpue = ncf_env.variables['cpue'][:] #Catch per unit effort
nobs = ncf_env.variables['nobs'][:] #Observation mask. 1 if observation at given c-square, 0 if none
time = ncf_env.variables["time"][:] #Time
lon  = ncf_env.variables["lon"][:] #Longitude
lat  = ncf_env.variables["lat"][:] #Latitude

msl  = normalized_netcdf_variable(ncf_env, 'msl',     (0,     1e10)) #Lufttryk
u10  = normalized_netcdf_variable(ncf_env, 'u10',     (-1000, 1000)) #10m øst vind
v10  = normalized_netcdf_variable(ncf_env, 'v10',     (-1000, 1000)) #10m vest vind
attn = normalized_netcdf_variable(ncf_env, 'attn',    (0,     1e10)) #lysdæmpning i vand
CHL  = normalized_netcdf_variable(ncf_env, 'CHL',     (0,     1e10)) # klorofyl - mg eller mu_g/L
mld  = normalized_netcdf_variable(ncf_env, 'karamld', (0,     12e3)) #Dybde af springlag
salt = normalized_netcdf_variable(ncf_env, 'vosaline',  (0,     100)) #salinitet
sst  = normalized_netcdf_variable(ncf_env, 'analysed_sst', (0, 500)) #overfladetemperatur
swh  = normalized_netcdf_variable(ncf_env, 'swh',     (0,     1000))#bølgehøjde
sst_front = normalized_netcdf_variable(ncf_env,  'sst_front',  (0,     1000)) #temperaturfrontindex
salt_front = normalized_netcdf_variable(ncf_env, 'salt_front', (0,     1000)) #salinitetsfrontindex
ncf_env.close()


#Could you do a sort of 3D PCA on variables to find which ones correlate to CPUE? 

cpue = np.where(cpue>0, np.log(cpue,where=cpue>0), 0) # log transform - needed?? Negative values are CPUE<1

data=np.stack((cpue,msl,u10,v10,attn,CHL,mld,salt,sst,swh,sst_front,salt_front),axis=0) #Data to be used as input
data_input_shape = np.shape(data)
#%% Resize by interpolation
#Resizing data - remember to resize back after prediction with network. 
#Resizing is done so aspect ratio between lat and lon are preserved - finding whole ints that match this restriction 
#These are: (29,25),(58,50),(87,75),(116,100),(145,125)
resize_lat=50
resize_lon=58

#Interpolation type. 
interpolation = cv2.INTER_AREA #Linear area interpolation. Works as linear 1D interpolation when applied to 1D array. https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
lat_resized = cv2.resize(lat.reshape(np.size(lat),1),(1,resize_lat),interpolation=interpolation).reshape(resize_lat)
lon_resized = cv2.resize(lon.reshape(np.size(lon),1),(1,resize_lon),interpolation=interpolation).reshape(resize_lon)

#Resizing parameters
cpue_resized = np.transpose(cv2.resize(np.transpose(cpue,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
msl_resized = np.transpose(cv2.resize(np.transpose(msl,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
u10_resized = np.transpose(cv2.resize(np.transpose(u10,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
v10_resized = np.transpose(cv2.resize(np.transpose(v10,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
attn_resized = np.transpose(cv2.resize(np.transpose(attn,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
CHL_resized = np.transpose(cv2.resize(np.transpose(CHL,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
mld_resized = np.transpose(cv2.resize(np.transpose(mld,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
salt_resized = np.transpose(cv2.resize(np.transpose(salt,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
sst_resized = np.transpose(cv2.resize(np.transpose(sst,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
swh_resized = np.transpose(cv2.resize(np.transpose(swh,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
sst_front_resized = np.transpose(cv2.resize(np.transpose(sst_front,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))
salt_front_resized = np.transpose(cv2.resize(np.transpose(salt_front,(1,2,0)),(resize_lon,resize_lat),interpolation=interpolation),(2,0,1))

#Have to re-stack all resized parameters, since OpenCV resize only takes up to 3D arrays
data_resized=np.stack((cpue_resized,msl_resized,u10_resized,v10_resized,attn_resized,CHL_resized,mld_resized,salt_resized,sst_resized,swh_resized,sst_front_resized,salt_front_resized),axis=0)
data_resized=np.transpose(data_resized,[1,2,3,0]) #Transposing the channels last


#for doing multi-channel network
split = [0.6,0.2,0.2] #the split of the data: training:validation:test
split = [int(len(time)*split[0]),int(len(time)*(split[0]+split[1]))]

#chosing years instead
#split_years = [2017,2018]
#split = [np.searchsorted(time,split_years[0]),np.searchsorted(time,split_years[1])]


train_data = data_resized[:split[0],:,:,:]
val_data = data_resized[split[0]:split[1],:,:,:]
test_data=data_resized[split[1]:,:,:,:]

time_train=time[:split[0]]
time_val=time[split[0]:split[1]]
time_test=time[split[1]:]

#For doing 1ch networks
train_cpue = train_data[0,:,:,:,np.newaxis]
val_cpue = val_data[0,:,:,:,np.newaxis]
test_cpue = test_data[0,:,:,:,np.newaxis]

#%% Visualize an area of a data type to use in training
data_format = cpue_resized
vararea=data_format[:,:,:]
#max and min values of the dataset
cpuemax = np.amax(cpue)
cpuemin = np.amin(cpue)
varmax = np.amax(vararea)
varmin = np.amin(vararea)
#colormap and normalization
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=cpuemin,vmax=cpuemax)
varnorm = mpl.colors.Normalize(vmin=varmin,vmax=varmax)
#margins for the map projection
ymargin = 0
xmargin = 0

#Setting up the figure
fig=plt.figure()
spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, figure=fig,width_ratios=[15,1])
#Axis with coastlines
ax2 = fig.add_subplot(spec[0],projection=ccrs.PlateCarree())
ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
#colorbar
cax = fig.add_subplot(spec[1]) 
#cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
#                                norm=norm,
#                                orientation='vertical')

varcb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=varnorm,
                                orientation='vertical')
cax.set_ylabel('Log10(CPUE)')


def onClick(event):
    #For pausing animation
    #From: https://stackoverflow.com/questions/16732379/stop-start-pause-in-python-matplotlib-animation
    global pause
    pause ^= True
fig.canvas.mpl_connect('button_press_event', onClick)

def animatepred(i):
    #Function to be repeated in animation
    
    ax2.clear()
    
    #visualize as contour
#    ax2.contourf(lon_resized, lat_resized, vararea[i,:,:], 10,
#             transform=ccrs.PlateCarree(),norm=varnorm,cmap=cmap)
    #visualize raster
    ax2.pcolormesh(lon_resized, lat_resized, vararea[i,:,:],
               transform=ccrs.PlateCarree(),norm=varnorm,cmap=cmap)

    ax2.coastlines(resolution='10m', color='black', linewidth=1)
    
    #ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black',linewidth=.5,facecolor=sns.xkcd_rgb['moss green']))
    
    ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
    ax2.set_title('CPUE predicted iter: %d'%(i))
    return ax2

anim1 = animation.FuncAnimation(fig,animatepred,frames=241,blit=False,interval=200)
plt.show()

#%% ------ create learning set ------
Seq_size=10
def to_sequence(seq_size,obs,dims=False,sample=False):
    #Here, observations are the 2D spatial data for each time point 
    #Returns a seq_size length sequence of the observations, with the immediate next observation as target
    #Sample is a tuple containing (latmin,latmax),(lonmin,lonmax)
    x = []
    xnew=[]
    y = []
    #obs.reshape((obs.shape[0],obs.shape[1]*obs.shape[2])) #reshaping into 2D observations. Not needed
    for i in range(np.size(obs,0)-seq_size): #Taking the total abount of observation points and subtracting the 
        x.append(obs[i:i+seq_size]) #training set
        y.append(obs[i+seq_size]) #value after training set.
    x=np.array(x)
    y=np.array(y)
    y=y[:,:,:,0,np.newaxis] #Chose only the CPUE as target. To have same dimensionality as the train sequence
    if isinstance(dims,tuple):
        #If dims is specified, only select these dims
        dimdict = {
                "cpue":0,
                "msl":1,
                "u10":2,
                "v10":3,
                "attn":4,
                "CHL":5,
                "mld":6,
                "salt":7,
                "sst":8,
                "swh":9,
                "sst_front":10,
                "salt_front":11,
                "all":[0,1,2,3,4,5,6,7,8,9,10,11] #edit this for doing custom, easier selections
                } 
        
        for channel in dims:
            #loop over all chosen dimensions
            if channel in dimdict:
                #check if channel name is in dictionary
                print(dimdict[channel])
                xnew.append(x[:,:,:,:,dimdict[channel]])
            else:
                print("channel:'%s' is not a valid parameter"%channel)
        x=np.array(xnew)
        x=np.transpose(x,[1,2,3,4,0])
    if isinstance(sample,tuple):
        #if there is a sample area specified, extract only sampled area
        x=x[:,:,sample[0][0]:sample[0][1],sample[1][0]:sample[1][1],:] #small part of data for testing purposes
        y=y[:,:,sample[0][0]:sample[0][1],sample[1][0]:sample[1][1],:]
    
    #TODO: Make sequences randomly picked from observations. This in order to learn more robustly
    return x,y
#dims=("cpue","msl","salt_front")
dims=("all")
x_train,target_train = to_sequence(Seq_size,train_data,dims=dims)
x_val,target_val = to_sequence(Seq_size,val_data,dims=dims)
x_test,target_test = to_sequence(Seq_size,test_data,dims=dims)
input_shape_simple=np.shape(x_train)[1:] #Shape is (batch size,lat,lon,channels). The number of samples does not need to be specified in Keras
output_shape_simple=np.shape(target_train)[1:]


#%% Another learning set - with target as ground truth shifted by 1 timestep
def to_sequence_shift(seq_size,obs,dims=False,sample=False):
    #Here, observations are the 2D spatial data for each time point
    #Returns a sequence of length seq_size and target data.
    #Target data is a sequence of the same length, shifted by 1 in a positive timestep
    #Sample is a tuple containing (latmin,latmax),(lonmin,lonmax) used to sample a specific area
    x = []
    xnew=[]
    y = []
    for i in range(np.size(obs,0)-seq_size): #Taking the total abount of observation points and subtracting the seq size 
        x.append(obs[i:i+seq_size]) #training set
        y.append(obs[(i+1):(i+seq_size+1)]) #values shifted by 1 in the number of frames
    x=np.array(x)
    y=np.array(y)
    y=y[:,:,:,:,0,np.newaxis] #Chose only the CPUE as target. To have same dimensionality as the train sequence
    if isinstance(dims,tuple):
        
        
        #If dims is specified, only select these dims
        dimdict = {
                "cpue":0,
                "msl":1,
                "u10":2,
                "v10":3,
                "attn":4,
                "CHL":5,
                "mld":6,
                "salt":7,
                "sst":8,
                "swh":9,
                "sst_front":10,
                "salt_front":11,
                "all":[0,1,2,3,4,5,6,7,8,9,10,11] #edit this for doing custom, easier selections
                } 
        
        for channel in dims:
            #loop over all chosen dimensions
            if channel in dimdict:
                #check if channel name is in dictionary
                print(dimdict[channel])
                xnew.append(x[:,:,:,:,dimdict[channel]])
            else:
                print("channel:'%s' is not a valid parameter"%channel)
        x=np.array(xnew)
        x=np.transpose(x,[1,2,3,4,0])
    if isinstance(sample,tuple):
        #if there is a sample area specified, extract only sampled area
        x=x[:,:,sample[0][0]:sample[0][1],sample[1][0]:sample[1][1],:] #small part of data for testing purposes
        y=y[:,:,sample[0][0]:sample[0][1],sample[1][0]:sample[1][1],:]
    
    #TODO: Make sequences randomly picked from observations. This in order to learn more robustly
    return x,y
#dims=("cpue","msl","salt_front")
dims=("all")
shifted_x_train,shifted_target_train = to_sequence_shift(Seq_size,train_data,dims=dims)
shifted_x_val,shifted_target_val = to_sequence_shift(Seq_size,val_data,dims=dims)
shifted_x_test,shifted_target_test = to_sequence_shift(Seq_size,val_data,dims=dims)

input_shape_simple=np.shape(shifted_x_train)[1:] #Shape is (batch size,lat,lon,channels). The number of samples does not need to be specified in Keras
output_shape_simple=np.shape(shifted_target_train)[1:]
  
#%% Make sure targets fit data - for debugging purposes
fig = plt.figure()
cmap = mpl.cm.jet
cmax = np.amax(cpue_resized)
cmin = np.amin(cpue_resized)
norm = mpl.colors.Normalize(vmin=cmin,vmax=cmax)
xmargin=0
ymargin=0

spec = mpl.gridspec.GridSpec(ncols=3,nrows=1,figure=fig,width_ratios=[15,15,1])
ax1 = fig.add_subplot(spec[0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(spec[1],projection=ccrs.PlateCarree())
cax = fig.add_subplot(spec[2])
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
ax1.coastlines(resolution='10m', color='black', linewidth=1)
ax1.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
ax2.coastlines(resolution='10m', color='black', linewidth=1)
ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
ax1.set_title("Last observation in next training")
ax2.set_title("Target")
cax.set_ylabel("log10(CPUE)")

i=29
ax1.contourf(lon_resized,lat_resized,x_val[1+i,9,:,:,0],100,transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)
ax2.contourf(lon_resized,lat_resized,target_val[0+i,:,:,0],100,transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)


#%% Fitting function for running all models

#%%
"""### Initializing neural nets ###"""


#%% Fitting model 1
"""
Model 1
Same low complexity network trained on different cost functions, with different optimizers and different number of epochs
To determine good parameters for simple ConvLSTM, to be applied to more complex architechtures. 
"""
epochs_model1=30

##
#Simplest model - mse loss
model1a=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1a.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1a")

# using custom rmse loss
model1b=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.custom_rmse) 
model1b.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1b")

## using weighted loss on zero predictions - 0.1*error for zero-predictions
model1c01=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.1)) 
model1c01.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1c01")
## using weighted loss on zero predictions - 0.2*error for zero-predictions
model1c02=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.2)) 
model1c02.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1c02")
# using weighted loss on zero predictions - 0.4*error for zero-predictions
model1c04=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.4)) 
model1c04.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1c04")
# using weighted loss on zero predictions - 0.6*error for zero-predictions
model1c06=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6)) 
model1c06.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1c06")
# using weighted loss on zero predictions - 0.8*error for zero-predictions
model1c08=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.8)) 
model1c08.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1c08")
#%% Loading already saved model weights 
model1a=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1a.load_weights(str(Path(working_path) / "model_weights/model1a/1/model.ckpt"))

model1b=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.custom_rmse) 
model1b.load_weights(str(Path(working_path) / "model_weights/model1b/1/model.ckpt"))

model1c01=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.1)) 
model1c01.load_weights(str(Path(working_path) / "model_weights/model1c01/1/model.ckpt"))

model1c02=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.2)) 
model1c02.load_weights(str(Path(working_path) / "model_weights/model1c02/1/model.ckpt"))

model1c04=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.4)) 
model1c04.load_weights(str(Path(working_path) / "model_weights/model1c04/1/model.ckpt"))

model1c06=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6)) 
model1c06.load_weights(str(Path(working_path) / "model_weights/model1c06/1/model.ckpt"))

model1c08=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.8)) 
model1c08.load_weights(str(Path(working_path) / "model_weights/model1c08/1/model.ckpt"))

#%% Evaluation of different weights
eval1a=Model_evaluate.evaluate(model1a,"mse loss",x_test,target_test)
eval1b=Model_evaluate.evaluate(model1b,"rmse loss",x_test,target_test)
eval1c01=Model_evaluate.evaluate(model1c01,"weighted loss 0.1",x_test,target_test)
eval1c02=Model_evaluate.evaluate(model1c02,"weighted loss 0.2",x_test,target_test)
eval1c04=Model_evaluate.evaluate(model1c04,"weighted loss 0.4",x_test,target_test)
eval1c06=Model_evaluate.evaluate(model1c06,"weighted loss 0.6",x_test,target_test)
eval1c08=Model_evaluate.evaluate(model1c08,"weighted loss 0.8",x_test,target_test)


#%% Plot values
#each eval array contains:loss,mse,mae,corr,shiftcor,diffcor,model name

plotdata = np.array([eval1a,eval1b,eval1c01,eval1c02,eval1c04,eval1c06,eval1c08])
plotdata = np.insert(plotdata,0,np.array(["Loss","MSE","MAE","Corrr","Shiftcorr","Diffcorr","Model Name"]),axis=0)
colors = mpl.cm.rainbow(np.linspace(0, 1, np.shape(plotdata)[0]))
plt.title("Evaluation parameters for same model, different cost functions")
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
for model,c in zip(plotdata[1:],colors):  
      # print(model[:-1])
      #print(c)
      plt.scatter(np.linspace(0,5,np.shape(model)[0]-1),model[:-1].astype(np.float),color=c,label=model[-1])
plt.legend()
plt.xticks(np.arange(6),plotdata[0,:-1])
plt.grid(ls="--")

#%% further model 1 parameter tuning
#Using 0.6 and more epochs
model1d=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6)) 
model1d.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1*2,working_path=working_path,model_name="model1d")

#Optimizers:
#Adam high learning rate
model1eAdam01=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6),optimizer="Adam",lr=0.01) 
model1eAdam01.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1eAdam0_01")

#Adam lower learning rate, more epochs
model1eAdam003=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6),optimizer="Adam",lr=0.003) 
model1eAdam003.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1*2,working_path=working_path,model_name="model1eAdam0_003")


#Adelta
model1eAdadelta=Model_lib.model_1(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.6),optimizer="Adadelta",) 
model1eAdadelta.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model1,working_path=working_path,model_name="model1eAdadelta")
#%% Load model weights
model1d=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1d.load_weights(str(Path(working_path) / "model_weights/model1d/1/model.ckpt"))
model1eAdam01=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1eAdam01.load_weights(str(Path(working_path) / "model_weights/model1eAdam0_01/1/model.ckpt"))
model1eAdam003=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1eAdam003.load_weights(str(Path(working_path) / "model_weights/model1eAdam0_003/1/model.ckpt"))
model1eAdadelta=Model_lib.model_1(input_shape_simple,output_shape_simple)
model1eAdadelta.load_weights(str(Path(working_path) / "model_weights/model1eAdadelta/1/model.ckpt"))
#%%
#Evaluate
eval1d=Model_evaluate.evaluate(model1d,"weighted loss 0.6, 2x epochs",x_test,target_test)
eval1eAdam01=Model_evaluate.evaluate(model1eAdam01,"Adam: lr 0.01",x_test,target_test)
eval1eAdam003=Model_evaluate.evaluate(model1eAdam003,"Adam: lr 0.003",x_test,target_test)
eval1eAdadelta=Model_evaluate.evaluate(model1eAdadelta,"Adadelta",x_test,target_test)

#%%
plotdata = np.array([eval1a,eval1b,eval1c01,eval1c02,eval1c04,eval1c06,eval1c08,eval1d])
plotdata = np.insert(plotdata,0,np.array(["Loss","MSE","MAE","Corrr","Shiftcorr","Diffcorr","Model Name"]),axis=0)
colors = mpl.cm.rainbow(np.linspace(0, 1, np.shape(plotdata)[0]))
plt.title("Evaluation parameters for same model, different cost functions")
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
for model,c in zip(plotdata[1:],colors):  
      # print(model[:-1])
      #print(c)
      plt.scatter(np.linspace(0,5,np.shape(model)[0]-1),model[:-1].astype(np.float),color=c,label=model[-1])
plt.legend()
plt.xticks(np.arange(6),plotdata[0,:-1])
plt.grid(ls="--")

#%% Optimizer comparison
plotdata = np.array([eval1eAdam01,eval1eAdam003,eval1eAdadelta])
plotdata = np.insert(plotdata,0,np.array(["Loss","MSE","MAE","Corrr","Shiftcorr","Diffcorr","Model Name"]),axis=0)
colors = mpl.cm.rainbow(np.linspace(0, 1, np.shape(plotdata)[0]))
plt.title("Evaluation parameters for same model, different optimizers")
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
for model,c in zip(plotdata[1:],colors):  
      # print(model[:-1])
      #print(c)
      plt.scatter(np.linspace(0,5,np.shape(model)[0]-1),model[:-1].astype(np.float),color=c,label=model[-1])
plt.legend()
plt.xticks(np.arange(6),plotdata[0,:-1])
plt.grid(ls="--")
#%%
"""
Model 2
"""
epochs_model2 = 30
model2=Model_lib.model_2(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model2.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model2,working_path=working_path)

#%%
"""
MODEL3:
"""
epochs_model3 = 30
model3 =Model_lib.model_3(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model3.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model3,working_path=working_path)


#%% Model 4
"""
MODEL 4:
Modification of keras ConvLSTM documentation
"""
epochs_model4 = 60
model4 =Model_lib.model_4(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model4.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model4,working_path=working_path)



#%% Model 5 - as functional style

        
"""
MODEL 5:
Network inspired by:
https://www.youtube.com/watch?v=MjFpgyWH-pk
Input dimensions need to be even for this architechture to work
"""
   
model5 = Model_lib.model_5(input_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
epochs_model5=80
model5.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model5,working_path=working_path,model_name="Model5")


#%% Model 6
"""
MODEL 6:
Network inspired by model 5 with noise and conv3D
Input dimensions need to be even for this architechture to work. for isloss_type=Model_lib.weightedRMSELoss(0.5)_int(input_dim/2**n), the higher n can be for this to be an int, the more feature-poolings can be done
"""
   
model6 = Model_lib.model_6(input_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
epochs_model6=80
model6.runmodel(x_train,target_train,x_val,target_val,batch_size=Seq_size,nb_epochs=epochs_model6,working_path=working_path,model_name="Model6")

#%% Load already saved models
model2=Model_lib.model_2(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model2.load_weights(str(Path(working_path) / "model_weights/model2/1/model.ckpt"))

model3=Model_lib.model_3(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model3.load_weights(str(Path(working_path) / "model_weights/model3/1/model.ckpt"))

model4=Model_lib.model_4(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model4.load_weights(str(Path(working_path) / "model_weights/model4/1/model.ckpt"))

model5=Model_lib.model_5(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model5.load_weights(str(Path(working_path) / "model_weights/model5/1/model.ckpt"))

model6=Model_lib.model_6(input_shape_simple,output_shape_simple,loss_type=Model_lib.weightedRMSELoss(0.5))
model6.load_weights(str(Path(working_path) / "model_weights/model6/1/model.ckpt"))


#%% Prediction tests

""" Single image prediction """
i = 20 #frame no
model = model2 #model to use


pred = model2.predict(x_test[i:i+1,:,:,:,:])
predtest=pred[0,:,:,0]
target = target_test[i,:,:,0]

cpuemax = np.amax(cpue_resized)
cpuemin = np.amin(cpue_resized)
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=cpuemin,vmax=cpuemax)
ymargin = .5
xmargin = 0
fig=plt.figure()
spec = mpl.gridspec.GridSpec(ncols=2, nrows=2, figure=fig,width_ratios=[15,1])
ax2 = fig.add_subplot(spec[0],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(spec[2],projection=ccrs.PlateCarree())
ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
ax3.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())

cax = fig.add_subplot(spec[1])
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
ax2.pcolormesh(lon_resized, lat_resized, pred[0,:,:,0], 100,
             transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)
ax3.pcolormesh(lon_resized, lat_resized, target, 100,
             transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)
ax2.coastlines(resolution='10m', color='black', linewidth=1)
ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
ax3.coastlines(resolution='10m', color='black', linewidth=1)
ax3.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())

ax2.set_title('Prediction')
ax3.set_title('Ground Truth')
cax.set_ylabel('Log10(CPUE)')
#%%MODEL EVALUATION
"""
Evaluation of model. Based on Zero Normalized Cross Correlation. A perfect correlation is 1, a perfect anticorrelation is -1
seeing the correlation over time - to check for actual prediction or just learning prev step
"""
eval2=Model_evaluate.evaluate(model2,"Model 2:Bi-directional ConvLSTM",x_test,target_test)
eval3=Model_evaluate.evaluate(model3,"Model 3:Deeper network",x_test,target_test)
eval4=Model_evaluate.evaluate(model4,"Model 4:Keras ConvLSTM doc",x_test,target_test)
eval5=Model_evaluate.evaluate(model5,"Model 5:maxpool+upscale",x_test,target_test)
eval6=Model_evaluate.evaluate(model6,"Model 6:maxpool+upscale+noise+3Dconvolution",x_test,target_test)
#%% Evaluation of complex models
# plotdata = np.array([eval2,eval3,eval4,eval5,eval6])
plotdata = np.array([eval1a,eval1b,eval1c01,eval1c02,eval1c04,eval1c06,eval1c08,eval1d,eval1eAdam01,eval1eAdam003,eval1eAdadelta,eval2,eval3,eval4,eval5,eval6])

plotdata = np.insert(plotdata,0,np.array(["Loss","MSE","MAE","Corr","Shifted corr","Total-Shifted corr","Model Name"]),axis=0)
colors = mpl.cm.rainbow(np.linspace(0, 1, np.shape(plotdata)[0]))
plt.title("Evaluation parameters for different models, evaluated on test set")
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
for model,c in zip(plotdata[1:],colors):  
      # print(model[:-1])
      #print(c)
      plt.scatter(np.linspace(0,4,np.shape(model)[0]-2),model[1:-1].astype(np.float),color=c,label=model[-1])
plt.legend(loc='upper left')
plt.xticks(np.arange(5),plotdata[0,1:-1])
plt.grid(ls="--")


#%% getweek

#%%
"""Model animation"""
model=model1eAdam003
# model_name="Model 6:Deep NN w/ maxpool+upscale+noise+3Dconvolution"
model_name="Model 1e: Adam w/ learning rate 0.003"

cpuemax = np.amax(cpue_resized)
cpuemin = np.amin(cpue_resized)
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=cpuemin,vmax=cpuemax)
ymargin = .5
xmargin = 0
fig=plt.figure()
spec = mpl.gridspec.GridSpec(ncols=2, nrows=2, figure=fig,width_ratios=[15,1])
#, [ax2,cax] = plt.subplots(1,2,gridspec_kw={"width_ratios":[15,1]})
ax2 = fig.add_subplot(spec[0],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(spec[2],projection=ccrs.PlateCarree())
ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
ax3.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())

cax = fig.add_subplot(spec[1])
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cax.set_ylabel('Log10(CPUE)')
fig.suptitle(model_name)



def animateboth(i):
    #xin=x_test[1:,:,:,:,:]+pred
    pred = model.predict(x_test[i:i+1,:,:,:,:])
    yr=int(time_test[i+10]) #year
    mo=int((time_test[i+10]-yr)*12)+1#month
    wk=int((time_test[i+10]-yr)*52)+1 #week
    predtest=pred[0,:,:,0]
    realtest=target_test[i,:,:,0]
    cpuemax = np.amax(cpue)
    cpuemin = np.amin(cpue)
    norm = mpl.colors.Normalize(vmin=cpuemin,vmax=cpuemax)
    ax2.clear()
    ax3.clear()
    ax2.pcolormesh(lon_resized, lat_resized, predtest,
             transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)
    ax2.coastlines(resolution='10m', color='black', linewidth=1)
    ax2.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
    ax2.set_title('CPUE predicted, week: %d %d'%(wk,yr))
    ax3.pcolormesh(lon_resized, lat_resized, realtest, 
                 transform=ccrs.PlateCarree(),norm=norm,cmap=cmap)
    ax3.coastlines(resolution='10m', color='black', linewidth=1)
    ax3.set_extent([np.amin(lon)-xmargin, np.amax(lon)+xmargin, np.amin(lat)-ymargin, np.amax(lat)+ymargin],ccrs.PlateCarree())
    ax3.set_title('CPUE real, week: %d %d'%(wk,yr))
    return ax2,ax3

anim3 = animation.FuncAnimation(fig,animateboth,frames=np.size(x_test,axis=0),interval=600,blit=False)
plt.show()

#For writing final animation
savedir = Path(working_path,"final_render/model1eAdam0_003.gif")
writergif = animation.PillowWriter(fps=1) 
anim3.save(savedir, writer=writergif)
