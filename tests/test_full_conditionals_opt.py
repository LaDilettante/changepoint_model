from changepoint.full_conditionals_opt import *

def test_Nk():
    Sn = np.array([1, 2, 1, 2, 2])
    ks = np.array([1, 2])

    true = np.array([2, 3])
    calculated = Nk(Sn)

    np.testing.assert_equal( true, calculated )

def test_Uk():
    Sn = np.array([1, 2, 1, 2, 2])
    Yn = np.array([3, 1, 4, 2, 6])

    true = np.array([7, 9])
    calculated = Uk(Yn, Sn)

    np.testing.assert_equal( true, calculated )