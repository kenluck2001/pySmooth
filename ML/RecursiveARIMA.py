#########################################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Recursive ARIMA
#########################################################################
from __future__ import division
import pandas as pd
import numpy as np
from utils import OnlineLinearRegression
from numpy import linalg as LA

class arma:
    """
    This implementation is inspired by the description in the papers. This is the arma proccess.

    Hamilton, J. D. (1994). Time series analysis. Chichester, United Kingdom: Princeton University Press.

    """

    def __init__( self, p=1, q=1  ):
        self.mLR = OnlineLinearRegression()
        self.window = None # store some data point in memory

        #set parameters
        self.p, self.q  = p, q


    def init( self, X):
        n, m = X.shape

        #white noise
        mean, std = 0, 1 
        self.error = np.random.normal(mean, std, size=self.q).reshape((1, self.q))

        mVal = max(self.p , self.q)
        
        if mVal + 1 > n :
            raise AssertionError("faulty matrix dimension as the error")

        y = X[-1].reshape((1, m)) #target prediction

        ind = n - 1
        pvec = X[ind-self.p: ind]
        qvec = self.error

        tvec = pvec.tolist()[0] +  qvec.tolist()[0]
        x = np.array ( tvec ).reshape((1, len(tvec)))


        self.mLR.update( x, y )  

        self.window = X



    def update ( self, x ):
        """
        x: is a row vector
        """

        #update the error

        theta = self.mLR.getA( )

        window = self.window

        n, m = window.shape

        mbias = np.ones((n, self.q))

        window = np.column_stack(( window, mbias))

        ypred = np.dot ( window, theta )

        error = self.window - ypred

        error = LA.norm(error, axis=1)

        ind = n 

        err = error[ind-self.q: ind]

        self.error =  np.array ( err ).reshape((1, self.q))


        #add element at back
        X = np.vstack(( self.window, x ))


        #remove element from front
        X = np.delete(X, (0), axis=0)

        self.window = X


    def predict ( self ):

        X = np.array (self.window )

        n, m = X.shape

        y = X[-1].reshape((1, m)) #target prediction

        ind = n - 1
        pvec = X[ind-self.p: ind]
        qvec = self.error

        tvec = pvec.tolist()[0] +  qvec.tolist()[0]
        yflat = np.array ( tvec ).reshape((1, len(tvec)))

        theta = self.mLR.getA( )

        #predict
        ypred = np.dot ( yflat, theta )

        return ypred




class RecursiveARIMA:

    def __init__( self, p=1, d=0, q=1  ):
        self.rma = arma( p=p+d, q=q )


    def init( self, X ):
        self.rma.init(X)


    def update ( self, x ):
        """
        x: is a row vector
        """
        self.rma.update ( x )


    def predict ( self ):
        return self.rma.predict ( )

