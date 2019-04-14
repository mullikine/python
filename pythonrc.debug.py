import shanepy
from shanepy import *

import pprint
import pprint as pp
import re
import matplotlib
import matplotlib as mp
import matplotlib as mpl


def print_callable(f):
    def decorated(*args, **kwargs):
        print f, type(f)
        return f(*args, **kwargs)
    return decorated


@print_callable
def multiply(x, y):
    print x * y
 
class A(object):
    @print_callable
    def foo(self):
        print 'foo() here'
