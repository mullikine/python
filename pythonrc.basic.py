import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os

import random

if (sys.version_info > (3, 0)):
    from past.builtins import execfile

def include(filename):
    if os.path.exists(filename): 
        execfile(filename)

# This doesn't appear to work.
# Use import?
#include("/var/smulliga/notes/issues/IMM-1090/et-test.py")
