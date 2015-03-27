import numpy as np
import scipy as sp
import scipy.stats as st

def P(m):
    '''
    Initialize a Markov transition matrix for the change point problem.

    Note that the probability of transitioning is only non-zero for moving from
    state k - 1 -> k and k -> k

    Args:
        m: The dimension of the transition matrix (i.e. the number of change points)
    Returns:
        the Markov transition matrix (m x m)
    '''
    return np.diag(np.concatenate([np.repeat(0.5, m - 1), [1]])) \
    + np.diag(np.repeat(0.5, m-1), k=1)

def S(n, m):
    '''Initialize a random vector of latent state variable. 

    This vector of latent state indicates which regime this time period is in.
    The location of change points (thus the location of regimes) is random.

    Args:
        n: The number of time periods
        m: The number of change points. We have m + 1 regimes, (1, ..., m + 1)
    Returns:
        a n-length vector of latent state variable s_t \in {1, ..., m + 1}
    '''

    # We use np.arange + 1 because our time and regime index starts at 1, not 0
    changepoint_locs = np.sort(np.random.choice(np.arange(2, n + 1), size=m, replace=False))
    regime_locs = np.concatenate([[1], changepoint_locs])

    return np.concatenate([np.repeat(np.arange(1, m + 1), np.diff(regime_locs)), \
                           np.repeat(m + 1, n - regime_locs[-1] + 1)])

