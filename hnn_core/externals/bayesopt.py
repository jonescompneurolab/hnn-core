"""Bayesian optimization according to:

Brochu, Cora, and de Freitas' tutorial at
http://haikufactory.com/files/bayopt.pdf

Adopted from http://atpassos.me/post/44900091837/bayesian-optimization
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Alexandre Passos <alexandre.tp@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Carolina Fernandez <cxf418@miami.edu>

import warnings
from sklearn import gaussian_process
import numpy as np

import scipy.stats as st


def expected_improvement(gp, best_y, x):
    """The expected improvement acquisition function. The equation is
    explained in Eq (3) of the tutorial.

    Parameters
    ----------
    gp : instance of GaussianProcessRegressor
        The GaussianProcessRegressor object.
    best_y : float
        Best objective value.
    x : ndarray
        Randomly distributed samples.

    Returns
    -------
    ndarray
        Object values corresponding to x.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, y_std = gp.predict(x, return_std=True)  # (n_samples, n_features)
    Z = (y - best_y) / (y_std + 1e-12)
    return (y - best_y) * st.norm.cdf(Z) + y_std * st.norm.pdf(Z)


def bayes_opt(f, initial_x, all_x, acquisition, max_iter=200,
              random_state=None):
    """The actual bayesian optimization function.

    Parameters
    ----------
    f : func
        The objective functions.
    initial_x : list
       Initial parameters.
    all_x : list of tuples
        Parameter constraints in solver-specific format.
    acquisition : func
        Acquisiton function we want to use to find query points.
    max_iter : int, optional
        Number of calls the optimizer makes. The default is 200.
    random_state : int, optional
        Random state of the gaussian process. The default is None.

    Returns
    -------
    best_x : list
        Optimized parameters.
    best_f : float
        Best objective value.

    """

    X, y = list(), list()

    # evaluate
    y.append(f(initial_x))
    X.append(initial_x)

    best_x = X[np.argmin(y)]
    best_f = y[np.argmin(y)]
    gp = gaussian_process.GaussianProcessRegressor(random_state=random_state)

    # draw samples from distribution
    all_x = np.random.uniform(low=[idx[0] for idx in all_x],
                              high=[idx[1] for idx in all_x],
                              size=(10000, len(all_x)))

    for i in range(max_iter):
        gp.fit(np.array(X), np.array(y))  # (n_samples, n_features)

        new_x = all_x[acquisition(gp, best_f, all_x).argmin()]  # lowest obj
        new_x = new_x.tolist()

        # evaluate
        new_f = f(new_x)

        if not np.isinf(new_f):
            X.append(new_x)
            y.append(new_f)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x

    return best_x, best_f
