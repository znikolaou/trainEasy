import libs
from tefcnetwork import *
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics

#Get data
mnist=tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest)=mnist.load_data()
xtrain, xtest=xtrain/255.0, xtest/255.0

#Specify network architecture
layersList=[784, 'bn', 128, {'dp':0.2}, 10]

#Build model, train and test
fcModel=TeFcNetwork(layersList)
fcModel.train(epochs=5, xtrain=xtrain, ytrain=ytrain, \
        loss=losses.SparseCategoricalCrossentropy(from_logits=True), \
        metrics=metrics.SparseCategoricalAccuracy())
fcModel.test(xtest, ytest)


