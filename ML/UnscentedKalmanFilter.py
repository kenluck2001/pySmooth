############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Unscented Kalman Filter
############################################################
from __future__ import division
import numpy as np
import math
from utils import OnlineLinearRegression



class UnscentedKalmanFilter:
    """
    This is a simplification of the algorithm description paper on the subject.

    [1] Wan , E. A., & Merwe, R. (2000). The unscented kalman filters for nonlinear estimation. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium, pp. 153-158.


    The paper made use of a feed forward network to predict the next measurement based on the immediate past measurement data. This implementation avoided the use of feed forward network to avoid the second system effects. Instead a recurvise linear regression was used to estimate the parameters.

    This paper made use of sigma matrix for each auto-coreelation between the present data and the past measurement. The mixing matrix results in a form of matrix factorization which was avoided in this implementation. In our solution, we have use vector as inputs to the recursive linear algorithm to avoid computational cost of matrix factorization.

        X is state
        Z is measurement


    [2] Julier, S. J. The scaled unscented transformation. Retrieved 06/15, 2017, from https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF

    [3] Galanos, K. Recursive least squares. Retrieved 06/15, 2017, from http://saba.kntu.ac.ir/eecd/people/aliyari/NN%20%20files/rls.pdf

    """


    def __init__( self ):
        '''
            accept the data in a matrix format
        '''
        self.olrState = OnlineLinearRegression() #state
        self.olrMeasurement = OnlineLinearRegression() #measurement

        self.x = None
        self.z = None

        self.xhat = None
        self.yhat = None
        self.P_yy = None
        self.P_xy = None
        self.Phatx = None


    def init( self, X, Z ):
        """
            perform initial train on a batch to estimate initial parameter.
            X: states, Z: measurement
        """
        #prepare z and z-1 for measurement
        for x, z in zip ( Z, Z[1:] ):
            xv, zv = x.reshape((1, len(x))), z.reshape((1, len(z)))
            self.olrMeasurement.update( xv, zv )            

        #prepare state measurement
        for xk_1, xk in zip ( Z, X ):
            x, y = xk_1.reshape((1, len(xk_1))), xk.reshape((1, len(xk)))       
            self.olrState.update( x, y )   


        self.x = np.mean ( X, axis=0 )[np.newaxis]
        self.z = np.mean ( Z, axis=0 )[np.newaxis]

        self.H = self.olrState.getA( )
        self.F = self.olrMeasurement.getA( )


    def Wm(self, X, k=0, alpha=0.0001):
        tmplist = []
        L = X.shape[1] #dimension of X
        start, end = 0, L
        lamda = ( alpha * alpha * ( L + k ) ) - L

        den = ( lamda + L )
        for ind in range ( start, end ):  
            if ind == start:
                if den < 0.00001:
                    tmp = 0.00001
                else:
                    tmp = lamda / den

                tmplist.append ( tmp )
            else:
                if den < 0.00001:
                    tmp = 0.00001
                else:
                    tmp = 0.5 / den

                tmplist.append ( tmp )

        tmparray = np.asarray([tmplist])

        indices = [0]*(L) 
        return tmparray[indices].T #convert a vector to a matrix and perform a transpose



    def Wc(self, X, k=0, alpha=0.001, beta=2):
        tmplist = []
        L = X.shape[1] #dimension of X
        start, end = 0, L
        lamda = ( alpha * alpha * ( L + k ) ) - L

        den = ( lamda + L )

        for ind in range ( start, end ):  
            if ind == start:

                tmp = 0
                if den < 0.00001:
                    tmp = 0.00001
                else:
                    tmp = lamda / den

                tmp =  tmp + ( 1 - (alpha * alpha) + beta ) 

                tmplist.append ( tmp )
            else:

                if den < 0.00001:
                    tmp = 0.00001
                else:
                    tmp = 0.5 / den

                tmplist.append ( tmp )

        tmparray = np.asarray([tmplist])

        indices = [0]*(L)
        return tmparray[indices].T #convert a vector to a matrix and perform a transpose


    def sigmaXMatrix ( self, X, k=0, alpha=0.0001 ):

        tmplist = []
        Xval = None
        L = X.shape[1] #dimension of X sigma vector
        start, end = 0, (2*L)
        Px = np.dot (X.T ,  X)
        lamda = ( alpha * alpha * ( L + k ) ) - L
        meanVec = X.reshape (( 1, L)) # for single data point no need to estimate the mean
        for ind in range ( start, end ):       
            if ind == start:
                tmplist.append ( meanVec.tolist()[0] )
            elif ind < ( end // 2 ) :
                tvec = ( L + lamda ) * Px[ind,]
                b = meanVec + np.power ( tvec, 0.5 )
                tmplist.append ( b.tolist()[0] )
            else:
                tvec = ( L + lamda ) * Px[(ind-L),]
                b = meanVec - np.power ( tvec, 0.5 )
                tmplist.append ( b.tolist()[0] )

        return np.asarray( tmplist ).T



    def update(self):
        '''
            update the parameter of the kalman filter
        '''
        #update prior

        zPos = np.dot ( self.z, self.F ) 

        X = self.sigmaXMatrix ( zPos ) # sigma matrix

        X = np.mat ( X )

        wm = np.mat ( self.Wm(X) )

        wc = np.mat ( self.Wc(X) )

        xhat =  X * wm

        tmm = (X - xhat)

        Phatx = wc * tmm.T * tmm

        yPos = np.dot ( self.z, self.H )

        yX = self.sigmaXMatrix ( yPos ) # sigma matrix

        wm_y = np.mat ( self.Wm(yX) )

        wc_y = np.mat ( self.Wc(yX) )

        Y = np.mat (yX)

        yhat =  Y * wm_y

        tmm = (Y - yhat)

        P_yy = wm_y * tmm.T * tmm

        tmmy = (Y - yhat)

        tmmx = (X - xhat)

        H = np.mat (self.H)


        P_xy = wc_y * tmmy.T * H.T * tmmx


        self.xhat = xhat

        self.yhat = yhat

        self.P_yy = P_yy

        self.P_xy = P_xy

        self.Phatx = Phatx



    def predict(self, x, z):
        """
            predict next state based on past state and measurement
        """
        #make the inverse of the matrix stable
        K = np.linalg.pinv ( np.nan_to_num(self.P_yy)  ) * self.P_xy  #kalman gain

        H = np.mat (self.H)

        nextstate =   z * H

        error = self.sigmaXMatrix ( x ) - self.sigmaXMatrix ( self.x )

        self.z = error * K * self.xhat.T
        self.z = np.mean ( self.z, axis=0 )[np.newaxis]

        self.Phatx = self.Phatx - (K.T * self.P_yy * K )


        self.olrMeasurement.update( self.z, x )    #( Z, Z[1:] )        
     
        self.olrState.update( self.z, x )   # ( Z, X )


        self.H = self.olrState.getA( )
        self.F = self.olrMeasurement.getA( )


        return nextstate



