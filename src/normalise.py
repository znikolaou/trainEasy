import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Normalise:
    """
    Class to normalise data

    Attributes:
        
    """
    def __init__(self, x, y):
        self._x=x
        self._y=y
        self._ns=np.shape(self._x)[0]
        self._nfx=np.shape(self._x)[1]
        self._nfy=np.shape(self._y)[1]

    def getZeroToOne(self):
        minx, maxx=self.getMinMax(self._x)
        miny, maxy=self.getMinMax(self._y)
        xnorm=np.zeros((self._ns, self._nfx))
        ynorm=np.zeros((self._ns, self._nfy))
        for i in range(self._nfx):
            xnorm[:,i]=self.__norm(self._x[:,i], minx[i], maxx[i])
        for i in range(self._nfy):
            ynorm[:,i]=self.__norm(self._y[:,i], miny[i], maxy[i])
        return xnorm, ynorm

    def getMinMax(self, x):
        ns=np.shape(x)[0]
        nf=np.shape(x)[1]
        minx=np.zeros(nf)
        maxx=np.zeros(nf)
        minx=[np.min(x[:,i]) for i in range(nf)]
        maxx=[np.max(x[:,i]) for i in range(nf)]
        return minx, maxx

    def __norm(self, x, minx, maxx):
        n=len(x)
        xnorm=np.zeros(n)
        if(minx==maxx):
            if(minx==0.0): return xnorm
            else:
                xnorm[:]=1.0
                return xnorm
        else:
            for i in range(n):
                xnorm[i]=(x[i]-minx)/(maxx-minx)
            return xnorm


