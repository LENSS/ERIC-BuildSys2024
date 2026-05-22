#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:50:53 2022

@author: tian
"""

# check how many rain vs non-rain data, day vs night time

import pandas as pd

fn = 'Output/RESULT_output_visual-only_temporal-filter_valAugSep_tstSepOct/val.csv'

data = pd.read_csv(fn)

print(data.info())

print(data.describe())