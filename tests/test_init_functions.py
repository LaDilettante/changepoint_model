from changepoint.init_functions import *
import pytest
import numpy as np

def test_P():
    assert((P(2) == np.array([[0.5, 0.5], [1]])).all)