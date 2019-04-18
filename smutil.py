# Importing libraries
import os

import time
import math

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

# plt.style.use('fivethirtyeight')
from pylab import *


import seaborn as sns

# sns.set(style="whitegrid")


import gc
from tqdm import tqdm


# import matplotlib.pyplot as plt
import matplotlib.style as pltstyle


font = {"family": "normal", "weight": "bold"}
matplotlib.rc("font", **font)


def create_directory(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory: {}/ Created ".format(dirName))
    else:
        print("Directory: {}/  already exists".format(dirName))


def line_break(headline="", pat="*", rep=20):
    print(f"{pat}" * rep + " | {} | ".format(headline) + f"{pat}" * rep)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))
