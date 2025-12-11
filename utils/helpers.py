import torch
import random
import numpy as np
import json
import os
import re
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def set_seed(seed = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
