import libs 
from tefcnetwork import *
import pytest

neuronsList=[10,100,50,5]
xtrain=np.ones((500,10,10))
ytrain=np.ones((500,5))
xtest=np.ones((50,10,10))
ytest=np.ones((50,5))

network=TeFcNetwork(neuronsList)

def testNoLayers(): assert network.getNoLayers()==len(neuronsList)

def testNoInputs(): assert network.getNoInputs()==neuronsList[0]

def testNoOutputs(): assert network.getNoOutputs()==neuronsList[-1]

def testGetDataFeatureSize(): assert network.getDataFeatureSize(xtrain)==np.prod(np.shape(xtrain)[1:])

def testGetDataSampleSize(): assert network.getDataSampleSize(xtrain)==np.shape(xtrain)[0]

def testUnknownLayerEntry(): 
    with pytest.raises(SystemExit) as e:
        network=TeFcNetwork([100, 'a'])
    assert e.type==SystemExit


