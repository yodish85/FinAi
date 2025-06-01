#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:37:47 2025

@author: Michele
"""

# Simple neural network model
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(3)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Simple dummy data
import numpy as np
X = np.random.random((10, 3))
y = np.random.random((10, 3))

# Train the model
model.fit(X, y, epochs=1)
