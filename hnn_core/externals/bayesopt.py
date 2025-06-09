"""Bayesian optimization according to:

Brochu, Cora, and de Freitas' tutorial at
http://haikufactory.com/files/bayopt.pdf

Adopted from http://atpassos.me/post/44900091837/bayesian-optimization
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Alexandre Passos <alexandre.tp@gmail.com>
#          Mainak Jas <mjas@mgh.harvard.edu>
#          Carolina Fernandez <cxf418@miami.edu>

import warnings
from sklearn import gaussian_process
import numpy as np

import scipy.stats as st


def expected_improvement(gp, best_f, all_x):
    """The expected improvement acquisition function. The equation is
    explained in Eq (3) of the tutorial.

    Parameters
    ----------
    gp : instance of GaussianProcessRegressor
        The GaussianProcessRegressor object.
    best_f : float
        Best objective value.
    all_x : ndarray
        Randomly distributed samples.

    Returns
    -------
    ndarray
        Object values corresponding to all_x.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # (n_samples, n_features)
        y, y_std = gp.predict(all_x, return_std=True)
    Z = (y - best_f) / (y_std + 1e-12)
    return (y - best_f) * st.norm.cdf(Z) + y_std * st.norm.pdf(Z)


def bayes_opt(func, x0, cons, acquisition, maxfun=200, debug=False, random_state=None):
    """The actual bayesian optimization function.

    Parameters
    ----------
    func : func
        The objective function.
    x0 : list
       Initial parameters.
    cons : list of tuples
        Parameter constraints in solver-specific format.
    acquisition : func
        Acquisition function we want to use to find query points.
    maxfun : int, optional
        Maximum number of function evaluations. The default is 200.
    debug : bool, optional
        The default is False.
    random_state : int, optional
        Random state of the GaussianProcessRegressor. The default is None.

    Returns
    -------
    best_x : list
        The argument that minimizes f.
    best_f : float
        The final objective value.
    """

    X, y = list(), list()

    # evaluate
    initial_f = func(x0)
    X.append(x0)
    y.append(initial_f)

    best_x = X[np.argmin(y)]
    best_f = y[np.argmin(y)]
    gp = gaussian_process.GaussianProcessRegressor(random_state=random_state)

    if debug:
        print("iter", -1, "best_x", best_x, best_f)

    for i in range(maxfun):
        # draw samples from distribution
        all_x = np.random.uniform(
            low=[idx[0] for idx in cons],
            high=[idx[1] for idx in cons],
            size=(10000, len(cons)),
        )

        gp.fit(np.array(X), np.array(y))  # (n_samples, n_features)

        # get new set of params
        new_x = all_x[acquisition(gp, best_f, all_x).argmin()]  # lowest obj
        new_x = new_x.tolist()

        # evaluate
        new_f = func(new_x)

        X.append(new_x)
        y.append(new_f)
        if new_f < best_f:
            best_f = new_f
            best_x = new_x

        if debug:
            print("iter", i, "best_x", best_x, best_f)

    return best_x, best_f


if __name__ == "__main__":
    from scipy.optimize import rosen

    opt_params, obj_vals = bayes_opt(
        rosen,
        [0.5, 0.6],
        [(-1, 1), (-1, 1)],
        expected_improvement,
        maxfun=200,
        random_state=1,
    )

    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)

    objs = rosen(np.vstack((X.ravel(), Y.ravel())))
