import numpy as np

def activacionSigmoide(x):
    return 1/(1+np.exp(-x))
def sigmoideDerivativo(x):
    return x *(1-x)
    