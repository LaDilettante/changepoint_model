from changepoint import init_functions
import pytest
import numpy as np

def test1():
    assert(P(2) == np.array([[0.5, 0.5], [1]]))