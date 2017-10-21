#########################################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An example of using the Recursive Linear Regression
#########################################################################
from __future__ import division
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../ML'))
from utils import OnlineLinearRegression


if __name__ == "__main__":
    X = np.random.rand(100,5)

    olr = OnlineLinearRegression()

    y = np.random.rand(1,5)
    x = np.random.rand(1,15)
    print x.shape, y.shape
    olr.update(x, y)


    print "A"
    print olr.getA( )
    print olr.getA( ).shape

    print "B"
    print olr.getB( )
    print olr.getB( ).shape

    """
    y = np.random.rand(1,5)
    x = np.random.rand(1,15)
    olr.update(x, y)

    print "A"
    print olr.getA( )

    print "B"
    print olr.getB( )

    y = np.random.rand(1,5)
    x = np.random.rand(1,15)
    olr.update(x, y)

    print "A"
    print olr.getA( )

    print "B"
    print olr.getB( )

    y = np.random.rand(1,5)
    x = np.random.rand(1,15)
    olr.update(x, y)

    print "A"
    print olr.getA( )

    print "B"
    print olr.getB( )


    print "mean y"
    print olr.getA().shape
    
    print "Noise Matrix"

    print olr.getCovarianceNoiseMatrix()
    """
