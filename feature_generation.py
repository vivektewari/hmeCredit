import pandas as pd
from dataExploration import distReports,plotGrabh
import numpy as np
from iv import iv_all,binning
import matplotlib.pyplot as plt
from funcs import crossVariable
import time
import seaborn as sns
start = time.time()

crossVariable(data[['CNT_FAM_MEMBERS', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORS