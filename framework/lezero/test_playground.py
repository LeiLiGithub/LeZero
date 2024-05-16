if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lezero import Variable, np

def test_variable_sum():
    a = Variable(np.array(1))
    b = Variable(np.array(2))
    print(a+b)

