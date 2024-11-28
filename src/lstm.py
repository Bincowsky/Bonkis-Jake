import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


df = pd.read_parquet('data/processed/2023_complete_pivot.parquet', parse_dates=['date'], index_col='date')