import random
import numpy as np
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm
from typing import Union, Tuple, List, Optional

#####################################################
# set the style of the plot
#####################################################
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.style.use('default')

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 10

rcParams['axes.unicode_minus'] = False
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

# rcParams['figure.constrained_layout.use'] = True
rcParams['figure.figsize'] = (2.8, 2.1)
rcParams['savefig.dpi'] = 600

rcParams['lines.linewidth'] = 1


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
#####################################################
# set the random seed
#####################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#####################################################
#####################################################

MSE = nn.MSELoss()
