from changepoint.init_functions import *

def test_P():
    np.testing.assert_array_equal( P(2), np.array([[0.5, 0.5], [0, 1]]) )

def test_S_have_correct_number_of_periods():
    # Test 100 times using random m and n
    for i in range(100):
        n = np.random.choice(np.arange(2, 101))
        m = np.random.choice(np.arange(1, n))    
        
        print n, m

        np.testing.assert_equal( len(S(n, m)), n )
        

def test_S_have_correct_number_of_regimes():
    # Test 100 times using random m and n
    for i in range(100):
        n = np.random.choice(np.arange(2, 101))
        m = np.random.choice(np.arange(1, n))    
        
        print n, m
        np.testing.assert_array_equal( np.unique(S(n, m)), np.arange(1, m + 2) )