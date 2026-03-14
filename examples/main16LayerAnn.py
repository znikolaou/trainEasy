import libs
from tefcnetwork import *
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics

#Specify network architecture
nLayers=2**4
layerList=[]
for i in range(nLayers):
    layerList.append(2**(nLayers-i))

#Build model
fcModel=TeFcNetwork(layerList)

