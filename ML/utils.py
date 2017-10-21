#########################################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Recursive Linear Regression
#########################################################################
from __future__ import division
import pandas as pd
import numpy as np


class OnlineLinearRegression:
    """
    This is based on the Recursive Linear regression classifier

    [1] Galanos, K. Recursive least squares. Retrieved 06/15, 2017, from http://saba.kntu.ac.ir/eecd/people/aliyari/NN%20%20files/rls.pdf

    """

    def __init__(self):
        '''
            accept the data in a matrix format
        '''
        self.error = None
        self.theta = None
        self.cov = None
        self.isTrained = False


    def update(self, x, y):
        '''
            accept in same format and dimension as data in constructor
        '''

        #set the prior
        self.priorY = y

        self.priorX = x

        xm = np.mat (x)
        ym = np.mat (y)


        all_zeros = not np.any(self.error)
        if all_zeros:
            self.error = np.zeros(y.shape)


        all_zeros = not np.any(self.theta)
        if all_zeros:
            #self.theta = ( (xm.T * xm ).I ) * xm.T *  ym 

            tmat = np.linalg.pinv( np.nan_to_num(xm.T * xm) ) #ensure stability
            self.theta = ( tmat ) * xm.T *  ym 
            self.isTrained = True


        all_zeros = not np.any(self.cov)
        if all_zeros:
            self.cov = np.dot ( x.T, x)



        if not self.isTrained:

            cov = np.mat ( self.cov )

            theta = np.mat ( xm * self.theta )

            self.error = ym - ( theta  )

            Im = np.mat ( np.eye (x.shape[1]) )


            self.cov = cov * np.mat( Im - ( ( xm.T * xm * cov ) / (1 + ( xm * cov * xm.T )  ) )  )
               
            self.theta = theta + ( self.cov * xm.T *  self.error  )   

            self.isTrained = False


    def getA(self):
        """
            outputs the A vector
        """
        return self.theta


    def getB(self):
        """
            outputs the noise or bias
        """
        return self.error


    def getCovarianceMatrix(self):
        """
            outputs the covariance matrix
        """
        #ypost = np.dot ( self.getA().T, self.priorX )

        theta = np.mat ( self.getA() )
        Xm = np.mat ( self.priorX )

        ypost = Xm * theta
        yprior = self.priorY
        error = ypost - yprior
        #error = error - np.mean ( error, axis = 0 )
        return np.dot ( error.T, error )



    def getCovarianceNoiseMatrix(self):
        """
            outputs the noise covariance matrix, R
        """
        return np.dot ( self.getB().T, self.getB() )




