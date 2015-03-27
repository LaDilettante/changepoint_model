import numpy as np
import scipy as sp
import scipy.stats as sp

def P(m):
    return np.diag(np.concatenate([np.repeat(0.5, m - 1), [1]])) \
    + np.diag(np.repeat(0.5, m-1), k=1)