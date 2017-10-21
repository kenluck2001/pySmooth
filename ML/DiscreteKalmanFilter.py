############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An implementation of Discrete Kalman Filter
############################################################
from __future__ import division
import numpy as np
from utils import OnlineLinearRegression

class DiscreteKalmanFilter:
    """
    This is based on the description in the paper


    Welch, G., & Bishop, G. An introduction to the kalman filter. Retrieved 06/15, 2017, from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf.


    The equations presented can be described as
    xk = Axk-a + Buk-1
    P = APk-1A.T + Q

    all the variable names match the meaning in the paper.

    x: state

    z: measurement


    Caveat: The algorithm only work with the dimension of X and Z are equal.
In practice, we can pad with zero and select the specific column.


    Q cannot be observed directly and as such cannot be observed but can be randomized as bad choice may still give good result.
    """

    def __init__( self ):

        self.A = None
        self.P = None
        self.Q = None

        self.H = None
        self.R = None

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


        self.A = self.olrState.getA( )
        self.P = self.olrState.getCovarianceMatrix( )

        #prepare z and x for measurement
        for x, z in zip ( X, Z ):
            a, b = x.reshape((1, len(x))), z.reshape((1, len(z)))
            self.olrMeasurement.update(a, b)            

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

        P = np.mat ( self.P )
        R = np.mat ( self.R )
        H = np.mat (self.H )

        z = np.mat ( self.z )
        x = np.mat ( self.x )

        tempM = ( H.T * P * H ) + R


        tmat = np.linalg.pinv( np.nan_to_num( tempM ) ) #ensure stability

        K = P * H * tmat

        
        self.x = x + (  ( z - ( x * H  ) ) * K.T  )
        
        I = np.mat ( np.eye( len( P ) ) )
        self.P = (I - ( K * H.T ) ) * P




    def predict(self, x, z):
        """
            predict next state based on past state and measurement
        """

        self.z = z
        nextstate =  np.dot ( x, self.A )

        #update posterior
        self.x = nextstate

        P = np.mat ( self.P )
        Q = np.mat ( self.Q  )
        A = np.mat (self.A )


        self.P = ( A * P * A.T ) + Q

        #ADDED DETAILS

        self.olrState.update( x, nextstate )
        self.A = self.olrState.getA( )

        self.olrMeasurement.update( x, z )  
        #self.Q = self.olrMeasurement.getCovarianceNoiseMatrix()
        n,m = self.A.shape
        self.Q  = np.random.rand(n, m)
        self.H = self.olrMeasurement.getA( )

        return nextstate


