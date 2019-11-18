"""
    Predict prob for velocity
    from 20 till 60 km/h.
"""

import numpy as np

# Time actions
actions = [-10, 0, 10]
measurements = [40, 50, 50]

# Probabilities
probs = np.zeros((4, 5))
probs[0] = np.asarray([0, 0.2, 0.6, 0.2, 0])

"""
    Iteration 1
"""

# Compute prob given action -10
for x in range(probs.shape[1]-1):
    probs[1][x] = 0.3 * probs[0][x] + 0.7 * probs[0][x+1]

# Handle last entry
probs[1][4] = 0.3 * probs[0][4]

# Compute prob given measurement 40
for x in range(probs.shape[1]):
    # Compute velocity
    vi = x * 10 + 20

    if 40 < (vi-10):
        probs[1][x] = 0 * probs[1][x]

    if 40 == (vi-10):
        probs[1][x] = 0.1 * probs[1][x]

    if 40 == vi:
        probs[1][x] = 0.7 * probs[1][x]

    if 40 == (vi+10):
        probs[1][x] = 0.2 * probs[1][x]

    if 40 > (vi+10):
        probs[1][x] = 0 * probs[1][x]

# Normalize
eta = np.sum(probs[1])
probs[1] = probs[1]/eta

"""
    Iteration 2
"""

# Compute prob given action -10
for x in range(probs.shape[1]):
    probs[2][x] = 1 * probs[1][x]

# Compute prob given measurement 40
for x in range(probs.shape[1]):
    # Compute velocity
    vi = x * 10 + 20

    if 50 < (vi-10):
        probs[2][x] = 0 * probs[2][x]

    if 50 == (vi-10):
        probs[2][x] = 0.1 * probs[2][x]

    if 50 == vi:
        probs[2][x] = 0.7 * probs[2][x]

    if 50 == (vi+10):
        probs[2][x] = 0.2 * probs[2][x]

    if 50 > (vi+10):
        probs[2][x] = 0 * probs[2][x]

# Normalize
eta = np.sum(probs[2])
probs[2] = probs[2]/eta

"""
    Iteration 3
"""

# Compute prob given action -10
for x in range(1, probs.shape[1]):
    probs[3][x] = 0.8 * probs[2][x-1] + 0.2 * probs[2][x]

# Handle first case
probs[3][0] = 0.2 * probs[2][0]

# Compute prob given measurement 40
for x in range(probs.shape[1]):
    # Compute velocity
    vi = x * 10 + 20

    if 50 < (vi-10):
        probs[3][x] = 0 * probs[3][x]

    if 50 == (vi-10):
        probs[3][x] = 0.1 * probs[3][x]

    if 50 == vi:
        probs[3][x] = 0.7 * probs[3][x]

    if 50 == (vi+10):
        probs[3][x] = 0.2 * probs[3][x]

    if 50 > (vi+10):
        probs[3][x] = 0 * probs[3][x]

# Normalize
eta = np.sum(probs[3])
probs[3] = probs[3]/eta

print(probs)
