#########################################################################
######  Name: Kenneth Emeka Odoh
######  Purpose: An example of using the Recursive ARIMA
#########################################################################
from __future__ import division
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('../ML'))
from RecursiveARIMA import RecursiveARIMA


if __name__ == "__main__":
    X = np.random.rand(10,5)
    recArimaObj = RecursiveARIMA(p=6, d=0, q=6)
    recArimaObj.init( X )


    for ind in range (100):
        x = np.random.rand(1,5)
     
        print "----------------------------"
        print "{}#".format (ind+1)
        print "----------------------------"
        print "##############################"
        recArimaObj.update ( x )
        print recArimaObj.predict( )







