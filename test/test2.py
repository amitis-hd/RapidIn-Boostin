import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from typing import Dict, Optional, Sequence
import argparse
import logging
import torch
import json
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tree_influence import RapidIn as rapidin
import torch.multiprocessing as mp
import random
import numpy as np



CONFIG_PATH = None

def main():
    

    rapidin.init_logging()

    random.seed(42)
    np.random.seed(42)

    infl = rapidin.calc_infl_mp()
    print("Finished")

    

if __name__ == "__main__":
    mp.set_start_method('spawn')
    # mp.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()
