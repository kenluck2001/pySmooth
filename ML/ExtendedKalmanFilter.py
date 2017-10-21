############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Extended Kalman Filter
############################################################
from __future__ import division
import numpy as np
import math
from utils import OnlineLinearRegression

def f(X):
    return np.power ( X, 2 ) - X


def h(X):
    return X / ( 1 - X )




def differentiate(func, Y, z):
    n,m = Y.shape
    Y1  = 0.01 * np.random.rand(n, m)
    res = func( Y )
    incrementres = func( Y+Y1 )

    deltaY = incrementres - res

    if Y.shape[0]==z.shape[1] or Y.shape[1]==z.shape[0]:
        deltaY = deltaY.T


    z = np.mean ( z, axis=0 )
    m = len ( z )
    z1  = 0.01 * np.random.rand(1, m)
    resz1 = func( z )
    incrementresz1 = func( z + z1 )

    deltaX = incrementresz1 - resz1 



    if Y.shape[1]==z.shape[0]:
        deltaX = deltaX.T

    result = deltaY / deltaX 

    return result


class ExtendedKalmanFilter:
    """
    This implementation is inspired by the description in the papers.

    [1] Welch, G., & Bishop, G. An introduction to the kalman filter. Retrieved 06/15, 2017, from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf.

    [2] Terejanu, G. A. Extended kalman filter tutorial. Retrieved 06/15, 2017, from https://www.cse.sc.edu/~terejanu/files/tutorialEKF.pdf


    The equations presented can be described as
    xk = f (xk-a, wk-1 )
    zk = h (xk, vk)


    x: state

    z: measurement

    all the variable names match the meaning in the paper.

    Q cannot be observed directly and as such cannot be observed but can be randomized as bad choice may still give good result.
    """

    def __init__( self ):

        self.A = None
        self.P = None
        self.Q = None

        self.H = None
        self.R = None

        self.W = None

        self.olrState = OnlineLinearRegression() #state
        self.olrMeasurement = OnlineLinearRegression() #measurement

        self.x = None
        self.z = None



    def init( self, X, Z ):
        """
            perform initial train on a batch to estimate initial parameter.
            X: states, Z: measurement
        """
        #prepare xk and xk_1 for state
        for xk_1, xk in zip ( X, X[1:] ):
            x, y = xk_1.reshape((1, len(xk_1))), xk.reshape((1, len(xk_1)))       
            self.olrState.update( x, y )    


        self.P = self.olrState.getCovarianceMatrix( )


        self.A = self.olrState.getA( )

        #prepare z and x for measurement
        for x, z in zip ( X, Z ):
            a, b = x.reshape((1, len(x))), z.reshape((1, len(z)))
            self.olrMeasurement.update( a, b )            

        self.H = self.olrMeasurement.getA( )
        self.R = self.olrMeasurement.getCovarianceMatrix( )


        n,m = self.A.shape
        self.Q  = np.random.rand(n, m)        


        self.x = np.mean ( X, axis=0 ).reshape( (1, X.shape[1]) )
        self.z = np.mean ( Z, axis=0 ).reshape( (1, Z.shape[1]) )



    def update(self):
        '''
            update the parameter of the kalman filter
        '''
        #update prior

        v = self.olrMeasurement.getCovarianceNoiseMatrix( )
        w = self.olrState.getCovarianceNoiseMatrix( )


        self.A = differentiate(f, self.A, self.x)


        P = np.mat ( self.P )

        R = np.mat ( self.R )

        W = np.mat ( differentiate(f, self.A, w) )

        H = np.mat ( differentiate(h, self.H, self.x) )

        V = np.mat ( differentiate(h, self.H, v) )


        z = np.mat ( self.z )
        x = np.mat ( self.x )


        val = int ( math.fabs ( V.shape[0] - R.shape[0] ) ) or int ( math.fabs ( V.shape[1] - R.shape[1] ) ) # diffierence in dimension

        
        zeros =  np.random.rand(1, V.shape[0])
        zeros = zeros.reshape ( (1, V.shape[0]) )

        for ind in range ( val ):
            V = np.concatenate((V, zeros.T), axis=1) #pad with zeros

        tmpM = ( H * P * H.T ) + ( V * R * V.T )
        tmat = np.linalg.pinv( np.nan_to_num(tmpM) )  #ensure stability
        K =  P * H.T * tmat
        
        self.x = x + (  ( z - h( x * H.T ) ) * K.T )
        
        I = np.mat ( np.eye( len( P ) ) )

        self.P = (I - ( K * H ) ) * P

        self.W = W


    def predict(self, x, z):
        """
            predict next state based on past state and measurement
        """

        self.z = z
        nextstate =  f ( np.dot ( x, self.A ) )

        #update posterior
        self.x = nextstate

        P = np.mat ( self.P )
        Q = np.mat ( self.Q  )
        A = np.mat (self.A )


        self.P = ( A * P * A.T ) + self.W * Q * self.W.T

        #extra details

        self.olrState.update( x, nextstate )
        self.A = self.olrState.getA( )

        #self.A = differentiate(f, self.A, x)

        self.olrMeasurement.update( x, z )  

        n,m = self.A.shape
        self.Q  = np.random.rand(n, m)
        self.H = self.olrMeasurement.getA( )

        return nextstate



