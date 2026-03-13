import numpy as np
import tensorflow as tf 
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers

class LayerKeys:
    dpKey='dp'
    batchNormKey='bn'
    normKey='n'

class FcNetwork(LayerKeys):
    """
    Class for building a general-structure fully-connected neural network. 

    Attributes: 
        layersList ([]): List of neurons or layer type for each layer.
                         Possible choices:
                           int           -> no of layer neurons.
                           '{'dp': 0.5}' -> dictionay of dropout layer.
                           'bn'          -> batch normalisation layer.
                           'n'           -> normalisation layer.
                         *Example: [100, {'dp':0.5}, 50, 'bn', 5]
        activation (tf.keras.activations.) : activation function. 
                
    Author: Z. Nikolaou (2026)
    """
    def __init__(self, layersList=[], activation=activations.relu):
        self._aFunc=activation
        self._layersList=layersList
        self._noLayers=self.getNoLayers()
        self._seqModel=tf.keras.models.Sequential()
        self._history=None
        self.__build()

    def getNoLayers(self): return len(self._layersList)

    def getNoInputs(self): return self._layersList[0]

    def getNoOutputs(self): return self._layersList[-1]

    def getDataFeatureSize(self, x): return np.prod(np.shape(x)[1:])

    def getDataSampleSize(self, x): return np.shape(x)[0]

    def __printDataShapes(self, xtrain, ytrain, xtest, ytest):
        print('Data shapes:')
        print('\txtrain:', np.shape(xtrain))
        print('\tytrain:', np.shape(ytrain))
        print('\txtest:', np.shape(xtest))
        print('\tytest:', np.shape(ytest))

    def __checkInputDataShapes(self, xtrain, xtest):
        if self.getNoInputs() != self.getDataFeatureSize(xtrain):
            print('Error: no inputs != no of train data inputs, terminating ...')
            exit()
        if self.getNoInputs() != self.getDataFeatureSize(xtest):
            print('Error: no inputs != no of test data inputs, terminating ...')
            exit()

    def __build(self):
        self._seqModel.add(tf.keras.layers.Flatten(input_shape=(self._layersList[0], 1)))  
        for entry in self._layersList[1:self._noLayers]: 
            if(isinstance(entry, dict)):
                if self.dpKey in entry.keys():
                    self._seqModel.add(tf.keras.layers.Dropout(rate=entry[self.dpKey]))
            elif(isinstance(entry, int)):
                self._seqModel.add(tf.keras.layers.Dense(entry, activation=self._aFunc))
            elif(entry==self.batchNormKey):
                self._seqModel.add(tf.keras.layers.BatchNormalization())
            elif(entry==self.normKey):
                self._seqModel.add(tf.keras.layers.Normalization())
            else:
                print('Error: entry not recognized ->', entry)
                print('Terminating ...')
                exit()
        print(self._seqModel.summary())
                
    def train(self, epochs=10, validationFraction=0.0, optimiser=optimizers.Adam(), loss=losses.mse, metrics=metrics.mse, \
            xtrain=None, ytrain=None, xtest=None, ytest=None):
        self.__checkInputDataShapes(xtrain, xtest)
        self.__printDataShapes(xtrain, ytrain, xtest, ytest)
        self._seqModel.compile(optimizer=optimiser, loss=loss, metrics=[metrics])
        self.history=self._seqModel.fit(xtrain, ytrain, validation_split=validationFraction, epochs=epochs)
        
    def test(self, xtest, ytest): 
        tst_loss, tst_accu=self._seqModel.evaluate(xtest, ytest, verbose=2)
        print('Test loss=', tst_loss)
        print('Test accuracy=', tst_accu)
        
    def predict(self, x): return self._seqModel.predict(x)
