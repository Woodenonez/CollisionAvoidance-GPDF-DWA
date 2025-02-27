import sys
import math
from timeit import default_timer as timer
from copy import copy, deepcopy
from typing import Optional, Union, Type, cast
from scipy import interpolate


import numpy as np
import matplotlib.pyplot as plt
import casadi as ca # type: ignore

from matplotlib.axes import Axes
