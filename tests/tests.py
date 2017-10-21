from __future__ import division
import numpy as np
import unittest
import sys
import os
sys.path.append(os.path.abspath('../ML'))

from DiscreteKalmanFilter import DiscreteKalmanFilter
from ExtendedKalmanFilter import ExtendedKalmanFilter
from RecursiveARIMA import RecursiveARIMA
from TimeDifference import TimeDifference
from UnscentedKalmanFilter import UnscentedKalmanFilter
from utils import OnlineLinearRegression


class DiscreteKalmanFilterTestCase(unittest.TestCase):
    """Tests for DiscreteKalmanFilter."""

    def setUp(self):
        X = np.random.rand(2,2)
        Z = np.random.rand(2,12)

        self.dkf = DiscreteKalmanFilter()

        self.dkf.init( X, Z ) #training phase



    def test_dimension(self):
        """Check for dimension"""
        for ind in range (100):
            self.dkf.update( )   

            x = np.random.rand(1,2)
            z = np.random.rand(1,12)
            self.assertEqual( self.dkf.predict( x, z ).shape, x.shape )



class ExtendedKalmanFilterTestCase(unittest.TestCase):
    """Tests for ExtendedKalmanFilter."""

    def setUp(self):
        X = np.random.rand(2,2)
        Z = np.random.rand(2,12)

        self.dkf = ExtendedKalmanFilter()

        self.dkf.init( X, Z ) #training phase



    def test_dimension(self):
        """Check for dimension"""
        for ind in range (100):
            self.dkf.update( )   

            x = np.random.rand(1,2)
            z = np.random.rand(1,12)
            self.assertEqual( self.dkf.predict( x, z ).shape, x.shape )



class RecursiveARIMATestCase(unittest.TestCase):
    """Tests for RecursiveARIMA."""

    def setUp(self):
        self.X = np.random.rand(100,5)


    def test_dimension(self):
        """Check for dimension"""
        for indx in range (1,7):
            for indy in range (1,7):
                recArimaObj = RecursiveARIMA(p=indx, q=indy)
                recArimaObj.init( self.X )


                for ind in range (100):
                    x = np.random.rand(1,5)
                    recArimaObj.update ( x )
                    y = recArimaObj.predict( )
                    self.assertEqual( y.shape, x.shape )



class TimeDifferenceTestCase(unittest.TestCase):
    """Tests for TimeDifference."""

    def setUp(self):
        self.X = np.random.rand(200,5)


    def test_dimension(self):
        """Check for dimension"""
        for d in range (3,7):

            tdObj = TimeDifference(d)

            #train a model
            tdObj.train( self.X )

            #predict on lag
            y = np.random.rand(d,5)

            ypred = tdObj.predict ( y )

            self.assertEqual( ypred.shape, (1, 5) )



class UnscentedKalmanFilterTestCase(unittest.TestCase):
    """Tests for TimeDifference."""

    def setUp(self):
        X = np.random.rand(2,2)
        Z = np.random.rand(2,15)

        self.dkf = UnscentedKalmanFilter()

        self.dkf.init( X, Z ) #training phase


    def test_dimension(self):
        """Check for dimension"""
        for ind in range (100):
            self.dkf.update( )   

            x = np.random.rand(1,2)
            z = np.random.rand(1,15)

            self.assertEqual( self.dkf.predict( x, z ).shape, x.shape )





class OnlineLinearRegressionTestCase(unittest.TestCase):
    """Tests for OnlineLinearRegression."""

    def setUp(self):
        self.olr = OnlineLinearRegression()


    def test_dimension(self):
        """Check for dimension"""

        y = np.random.rand(1,5)
        x = np.random.rand(1,15)

        self.olr.update(x, y)

        self.assertEqual( self.olr.getA( ).shape, (15, 5) )

        self.assertEqual( self.olr.getB( ).shape, (1, 5) )


    def test_nulls(self):
        """Check for NANs"""

        y = np.random.rand(1,5)
        x = np.random.rand(1,15)

        self.olr.update(x, y)

        #check if inf exist in the matrix
        self.assertFalse( np.isnan( np.sum( np.sum(self.olr.getA( )) ) ) )
        self.assertFalse( np.isnan( np.sum( np.sum(self.olr.getB( )) ) ) )


    def test_infs(self):
        """Check for dimension"""

        y = np.random.rand(1,5)
        x = np.random.rand(1,15)

        self.olr.update(x, y)

        #check if NAN exist in the matrix
        self.assertFalse( np.isinf( np.sum( np.sum(self.olr.getA( )) ) ) )
        self.assertFalse( np.isinf( np.sum( np.sum(self.olr.getB( )) ) ) )


 
if __name__ == '__main__':
    unittest.main()


