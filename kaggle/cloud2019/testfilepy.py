# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:11:27 2019

@author: hmnsh
"""

import numpy as np
x = np.zeros((2,3,4))
ap = np.zeros((1,3,4))
y = np.append(x,ap, axis=0)