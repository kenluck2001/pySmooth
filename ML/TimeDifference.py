############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Time Difference Model
############################################################
from __future__ import division
import numpy as np
from utils import OnlineLinearRegression


class TimeDifference:
    """
    This implementation is inspired by the description in the papers.

    Hamilton, J. D. (1994). Time series analysis. Chichester, United Kingdom: Princeton University Press.

    """
    def __init__( self, d=1 ):
        self.mLR = OnlineLinearRegression()
        self.n, self.m, self.d = 0, 0 ,0
        self.d = d

    def train( self, X ):
        n, m = X.shape

        #set parameters
        self.n, self.m  = n, m

        for indx in range( 0,n-self.d ):
            for ind in range( indx, n, self.d ):

                if indx != ind:
                    continue

                beg = ind
                end = ind + self.d

                
                if end >= n:
                    end = n - 1

                if beg == end:
                    continue

                if end < ( n - 1 ):
                    x, y =  X[beg:end,].flatten().reshape((1, self.d*m)), X[ (end + 1),]
                    self.mLR.update( x, y )  


    def predict ( self, y ):

        yflat = y.flatten().reshape((1, self.m * self.d))

        theta = self.mLR.getA( )
        #predict
        ypred = np.dot ( yflat, theta )

        return ypred

