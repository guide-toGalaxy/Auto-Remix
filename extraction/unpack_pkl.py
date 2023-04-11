import pickle
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)


with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
    
    print(data)
