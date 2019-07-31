from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from distutils.dir_util import copy_tree
import sys
import glob
import pickle
import pandas as pd
import csv
import zipfile
import json
import multiprocessing as mp
from multiprocessing import Pool
from collections import OrderedDict
import itertools
from timeit import default_timer as timer
import shutil
import inspect
import numbers
import copy
import collections
from torch.nn.utils.rnn import *
from torch.nn.parallel.data_parallel import data_parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import matplotlib
import cv2
import PIL
import random
import numpy as np
import math
import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('/lib', ''))
IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# numerical libs
matplotlib.use('TkAgg')
# matplotlib.use('WXAgg')
# matplotlib.use('Qt4Agg')
# matplotlib.use('Qt5Agg') #Qt4Agg
print('matplotlib.get_backend : ', matplotlib.get_backend())
# print(matplotlib.__version__)


# torch libs


# std libs

#from pprintpp import pprint, pformat


# constant #
PI = np.pi
INF = np.inf
EPS = 1e-12
