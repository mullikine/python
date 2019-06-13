# This should not be the default pythonrc file. It's too slow and slows down
# emacs.

import sys

import pip

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])

try:
    import torch
    import torch as tc
except:
    pass

#  py i dopamine-rl

# appears stable enough for xpti
try:
    import_or_install("dopamine")
    #  import dopamine
except:
    pass

import mock

# minisom
# minimalistic, Numpy based implementation of the Self-Organizing Maps

try:
    from minisom import MiniSom
    #  som = MiniSom(6, 6, 4, sigma=0.5, learning_rate=0.5)
    #  som.train_random(data, 100)
except:
    pass

try:
    import fire # generate cli from python object
except:
    pass

# This is broken
# import edward

# This is my own module that will wrap every exception in a try catch
import mydebug

if (sys.version_info < (3, 0)):
    import macropy as mp


try:
    import moa as moa_bio
except:
    pass

from slugify import slugify

# This will allow me to quote commands.
import shlex

try:
    import pandas_gbq as pbq
except:
    pass

# OpenAI gym
import gym

import_or_install("github")

import github as gh
from github import Github

import fastai as fai

import unicodedata
import shanepy
from shanepy import *

import sklearn as sk
from sklearn.preprocessing import scale
#  vim +/"data = scale(data)" "$HOME$MYGIT/FlorianMuellerklein/Machine-Learning/MultiLayerPerceptron.py"

#  agi libfuzzy-dev
if (sys.version_info > (3, 0)):
    # It's not installing for python2
    import ssdeep
    import ssdeep as ss

#  import pydeep

import libtmux
ts = libtmux.Server()

# I have to start building more python libraries
# Build classes and things. Work towards tensorflow and spacy. Build things
# together. Learn high level concepts.

#  tss_name = ts.cmd("display-message", "-p", "#S").stdout[0]
#  tss = ts.find_where({"session_name": tss_name})


# Try to import everything

# Should probably try to get pygazebo importing

import os

#  , dirname
from os import access, chmod as _chmod, environ, getcwd, getuid, linesep, link, listdir, lstat, lstat as os_lstat, makedirs, mkdir, open as os_open, path, path as opath, path as ospath, path as p, readlink, rename, sep, stat, stat as os_stat, symlink, unlink, utime

import pdb
# This actually imports pdb++

from pdb import set_trace as bp

from os.path import join as path_join

import sys

#  str(os.path.basename(os.getcwd()))

import errno

#  MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. mujoco-py allows using MuJoCo from Python 3.
#  pip-install.sh mujoco_py
import gym

import shanepy
# Then can call shanepy.bash("ns hi")

from shanepy import *
# DISCARD this still doesn't import some things, such as function pointers
# I just didn't run setup install for shanepy

# Now can call bash("ns hi")

# This is old
#  from fuse import FUSE, FuseOSError, Operations

try:
    # import common if it exists in the current folder
    import common
except Exception:
    pass

import itertools
import itertools as it
from itertools import *

import re

if (sys.version_info > (3, 0)):
    from urllib.parse import *

#  import pygraphviz

# Unload matplotlib
# Does not work
# modules = []
# for module in sys.modules:
#     if module.startswith('matplotlib'):
#         modules.append(module)
#
# for module in modules:
#     sys.modules.pop(module)

import matplotlib
import matplotlib as mp
import matplotlib as mpl
import matplotlib.pyplot as plt

#  mpl.use('TkAgg', force=True)
#  mpl.use('TkCairo', force=True)
# Experimental
#  mpl.switch_backend('TkCairo')

# Not using the correct backend
import seaborn as sns

# Correct
#  print (matplotlib.rcParams['backend'])
# Correct
#  mpl.get_backend()

#  import logstash
#  import logstash as ls

import xml

import problog
import problog as pl
from problog.program import PrologString
from problog import get_evaluatable
from problog.evaluator import SemiringLogProbability
from problog.logic import Term, Constant


import Cython
import Cython as cy
import fasttext
import fasttext as ft



# Python streaming
import riko


import apscheduler


import sqlparse

#  import socket
import base64

if (sys.version_info > (3, 0)):
    from past.builtins import execfile

def source_file(fp):
    """Directly includes python source code."""

    sys.stdout = open(os.devnull, 'w')
    execfile(fp)
    sys.stdout = sys.__stdout__

if (sys.version_info > (3, 0)):

    #  /var/smulliga/notes/ws/tensorflow/setup/libcudnn.so.5.txt
    import tensorflow
    import tensorflow as tf

    ## Outdated? I think I only needed to reinstall tensorflow
    from tensorflow.python.framework import dtypes
    from tensorflow.contrib import learn as tflearn
    from tensorflow.contrib import layers as tflayers

    import tensorflow_probability as tfp
    import tensorflow_text as tft
    import tensorflow_text as text # The standard import

    # This is frustratingly broken
    # import tensorflow_datasets as tfds
    # this happen because you are using python 3 rather than python 3.7. Just install pip37/python37 to overcome such issue.

    tf.logging.set_verbosity(tf.logging.INFO)

    # This died
    #  import tensorflow_hub as hub

    import keras
    import keras as ks
    from keras import backend as K

    import pyspark
    import pyspark as ps
    import smartframes
    import smartframes as sf
    import tensorflowonspark
    import tensorflowonspark as tfs

    import kibana
    import kibana as ki

    import urlparse3

    import twisted
    import twisted as tw

    import celery
    import celery as ce

import re

import elasticsearch
import elasticsearch as es

# import all the python from this directory
# Although the code is fine, this doesn't always work actually import anything
try:
    #dirmodules = [re.sub(r"(.*)\.py$",r"\1", f) for f in os.listdir('.') if re.match(r'.*\.py$', f)]
    dirmodules = [re.sub(r"(.*)\.py$",r"\1", f) for f in os.listdir(os.getcwd()) if re.match(r'.*\.py$', f)]
    modules = map(__import__, dirmodules)
except:
    pass


# the vim module is only accessible when called from vim
# Otherwise, these is vimmock
#  import vim

# http://www.scipy-lectures.org/advanced/sympy.html#differentiation
import sympy
import sympy as sm

try:
    import parsy
except:
    pass

import sqlalchemy
import sqlalchemy as sa
import spacy
import spacy as sy
from nltk.stem.wordnet import WordNetLemmatizer
#  from spacy.en import English
import nltk
import nltk as nl
import argparse
import argparse as ap
import bisect
import bisect as bi
import calendar
import calendar as cl
import collections
import collections as co
import configparser
import configparser as cp
import distutils
import distutils as du
import errno
#  import exceptions
import fileinput
import fnmatch
#import formatter
#import formatter as fo
import fractions
import fractions as fr
import functools
import functools as fn
import getopt
import glob
import hashlib
import heapq
import io
import json
import json as jn
import logging
import logging as lg
import math
import math as m
import mimetypes
import mimetypes as mt
import os
import pickle
import pickletools
import pipes
import platform
import pprint
import pprint as pp
import pydoc
import pydoc as py
#  import pyqcy
import random
#  import repr
import reprlib
import setuptools
import setuptools as st
import shutil
import string
import sys
import tempfile
import tempfile as te
import time
import timeit
import urllib
#  import urllib2
import uuid
import weakref

import datetime
import datetime as dt
from datetime import *
import requests
import requests as rq
import unittest
import unittest as ut

import autograd.numpy as numpy
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

# def tanh(x):                 # Define a function
#     y = np.exp(-2.0 * x)
#     return (1.0 - y) / (1.0 + y)

# grad_tanh = grad(tanh)       # Obtain its gradient function
# grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
# (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences

# import numpy
# import numpy as np

import scipy
import scipy as sp
import pandas
import pandas as pd
import jinja2
import jinja2 as j2
import jenkinsapi
import jenkinsapi as ja
import jenkinsapi
import jenkinsapi as jk
import jenkinsapi
import jenkinsapi as je

import scipy.spatial.distance as dist

import google
import wolframalpha

import math
from xml.dom import minidom


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


try:
    import numpy as np
except ImportError:
    pass

try:
    import xml.etree.ElementTree as ET
    import xml.etree.ElementTree as et
except ImportError:
    pass

# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
# DEPRECATED
# Convenience for importing absolute paths
# Also means you don't need to comply to python's file naming conventions
#  import imp
#  foo = imp.load_source('et-test', '/var/smulliga/notes/issues/IMM-1090/et-test.py')

#  sys.path.append('/var/smulliga/notes/issues/IMM-1090/et-test.py')



def FindRow(value, column):
    try:
        n = next((i, cell) for i, cell in enumerate(column) if cell == value)
        return n
    except Exception as ex:
        return 0


# pretty xml
# This breaks
#  import lxml.etree as etree

# x = etree.parse("filename")
#  print etree.tostring(etree.fromstring(ET.tostring(rootNode)), pretty_print=True)

def normpath(p):
    return os.path.abspath(os.path.realpath(os.path.normpath(p)))


# this is for ipython only
# Lets you do this:
# import re
# %help re
try:
    from IPython.core import page
    from IPython.core.magic import register_line_magic

    @register_line_magic('help')
    def magic_help(s):
        """Retrieve the pydoc help for an object and display it through a pager.

        See http://stackoverflow.com/a/1176180/1410871
        """
        import pydoc
        import sys
        obj = reduce(getattr, s.split("."), sys.modules[__name__])
        page.page(pydoc.plain(pydoc.render_doc(obj)))
    del magic_help
except:
    pass

import yaml
#  six - Utilities for writing code that runs on Python 2 and 3
import six

import mock
import wheel
import google.protobuf
import google.protobuf as pb
# werkzeug makes web development easier, apparently
import werkzeug
import werkzeug as wz
import funcsigs
import pbr




#  Add useful things to the basic python interpreter
import readline

try:
    import rlcompleter
except:
    pass

import atexit

# history_file = os.path.join(os.environ['HOME'], '.python_history')
# try:
#     readline.read_history_file(history_file)
# except IOError:
#     pass
# readline.parse_and_bind("tab: complete")
# readline.set_history_length(1000)
# atexit.register(readline.write_history_file, history_file)
# del readline, rlcompleter, atexit, history_file


sys.path.append("/var/smulliga/notes2017/ws/python/")
import importdir

try:
    importdir.do(os.getcwd(), globals())
except Exception as ex:
    pass




import trace

tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=0,
    count=1)

#tracer.run('main()')
#
## make a report, placing output in the current directory
#r = tracer.results()
#r.write_results(show_missing=True, coverdir=".")


df1=o("/home/shane/var/smulliga/source/git/tensorflow/tensorflow/contrib/learn/python/learn/datasets/data/boston_house_prices.csv")
#  df2=o("/var/smulliga/notes/issues/IMM-1145/rm48_DataConfig_Utility-cdid-mie.csv").rename(columns={'Cdid':'CDID'})
df2=o("/home/shane/var/smulliga/source/git/visidata/sample_data/StatusPR.csv").rename(columns={'Cdid':'CDID', 'MenuItemElement':'ElementName'})
df=df1
# This is so I start with a dataframe to play around with.
s2=df2.var()
#s=s2
# And a series

def map_funcs(obj, func_list):
    return [func(obj) for func in func_list]


#  from countminsketch import CountMinSketch
#  sketch = CountMinSketch(1000, 10)  # table size=1000, hash functions=10
#  sketch.add("oh yeah")
#  sketch.add(tuple())
#  sketch.add(1, value=123)
#  print sketch["oh yeah"]       # prints 1
#  print sketch[tuple()]         # prints 1
#  print sketch[1]               # prints 123
#  print sketch["non-existent"]  # prints 0

# Memory-efficient Count-Min Sketch Counter (based on Madoka C++ library)
#  https://github.com/ikegami-yukino/madoka-python

# Try out madoka then

import madoka
sketch = madoka.Sketch()

# I want python version detection though. But I should just start building
# things with python. I should not do this with c++. I should slap together
# python things for no reason though.

from langdetect import detect
# detect("War doesn't show who's right, just who's left.")
# detect("Ein, zwei, drei, vier")


# Graph theory
import networkx as nx


import dataset as ds

#  http://hypertools.readthedocs.io/en/latest/auto_examples/precog.html#sphx-glr-auto-examples-precog-py
import hypertools as hyp

#  hyp.plot(list_of_arrays, '.', group=list_of_labels)
#  hyp.plot(list_of_arrays, align='hyper')
#  hyp.plot(array, '.', n_clusters=10)
#  hyp.tools.describe(list_of_arrays, reduce='PCA', max_dims=14)


# pillow
# This is not the way: "import pillow as plw". This is:
from PIL import Image


#  py i google-cloud-language
from google.cloud import language
#from google.cloud.language import enums
#from google.cloud.language import types


# py i google-cloud-pubsub
from google.cloud import pubsub

# py i google-cloud-vision
from google.cloud import vision

# py i google-cloud-bigtable
from google.cloud import bigtable


#  import autokeras # this is causing problems

#  import bowler

import sys
def GetPythonVersion():
    print(sys.version)
    return sys.version

def list_add(mylist, x):
    """Extend list if list, append to list if element"""

    mylist.extend( x if type(x) == list else [x] )

from googlesearch import search

from github import Github

from tabulate import tabulate
