import cPickle
import os
import os.path
import copy_reg
import types
import multiprocessing

import numpy as np
from utils import OnlineLinearRegression
from DiscreteKalmanFilter import DiscreteKalmanFilter

#handling the pickling object
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"



class FileManager:
    """
        This allows for saving a ML model to be loaded at a later time.
    """

    def __init__( self, file_name ):
        self.file_name = file_name

        modelDir = dir_path + "model/"

        #check if directory exist, otherwise create the 
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)

        self.fileNameWithPath = modelDir + self.file_name



    def save (self, classifier):
        '''
            write to file
        '''
        with open(self.fileNameWithPath, 'wb') as fout:
            cPickle.dump(classifier, fout)
        print 'model written to: ' + self.file_name



    def load (self):
        """
            load an existing file
        """
        ispresent =  os.path.exists(self.fileNameWithPath)

        if ispresent:
            module = cPickle.load( open( self.fileNameWithPath ) ) #get the model
            return module

        raise FileNotFoundError ("Attempting to load a module that was not saved!!!!")


if __name__ == "__main__":
    filename = "linear.pkl"
    mObject = OnlineLinearRegression() #state

    fObject = FileManager(filename)

    fObject.save (mObject) #save a model

    model = fObject.load( ) #load a model

    X = np.random.rand(100,2)

    for xk_1, xk in zip ( X, X[1:] ):
        x, y = xk_1.reshape((1, len(xk_1))), xk.reshape((1, len(xk_1)))       
        model.update( x, y ) 


    print model.getA( )

    fObject.save (model)

    ########################

    filename = "discrete.pkl"
    X = np.random.rand(2,2)
    Z = np.random.rand(2,2)

    dkf = DiscreteKalmanFilter()

    dkf.init( X, Z ) #training phase


    fObject = FileManager(filename)

    fObject.save (dkf) #save a model

    dkf = fObject.load( ) 
    for ind in range (100):
        dkf.update( )   

        x = np.random.rand(1,2)
        z = np.random.rand(1,2)

        print dkf.predict( x, z )



