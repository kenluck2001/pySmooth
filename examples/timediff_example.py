############################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An example of using the Time Difference Model
############################################################
from __future__ import division
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('../ML'))
from TimeDifference import TimeDifference


if __name__ == "__main__":

    X = np.random.rand(200,4)
    d = 5

    tdObj = TimeDifference(d)

    #train a model
    tdObj.train( X )

    #predict on lag
    y = np.random.rand(5,4)

    ypred = tdObj.predict ( y )

    print ypred

