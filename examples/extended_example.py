############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An example of using the Extended Kalman Filter
############################################################
from __future__ import division
import numpy as np
import math

import sys
import os
sys.path.append(os.path.abspath('../ML'))
from ExtendedKalmanFilter import ExtendedKalmanFilter


if __name__ == "__main__":

    X = np.random.rand(2,2)
    Z = np.random.rand(2,15)

    dkf = ExtendedKalmanFilter()

    dkf.init( X, Z ) #training phase


    for ind in range (100):
        dkf.update( )   

        x = np.random.rand(1,2)
        z = np.random.rand(1,15)
     
        print "----------------------------"
        print "{}#  {}, {}".format (ind+1, x,z)
        print "----------------------------"
        print "##############################"
        print dkf.predict( x, z )


