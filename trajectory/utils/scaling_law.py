# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.optimize import curve_fit

def fit_power_law(S, L):
    def power_law(S, alpha, beta, C):
        return C + (beta / S)**alpha

    params, covariance = curve_fit(power_law, S, L)
    alpha, beta, C = params

    def fitted_power_law(S):
        return power_law(S, alpha, beta, C)

    return fitted_power_law