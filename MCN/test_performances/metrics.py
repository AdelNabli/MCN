import numpy as np

def opt_gap(list_exact, list_approx):

    """Function used to compute the optimality gap:
    mean((val_exact - val_approx) / val_exact)"""

    vals_exact = np.array(list_exact)
    vals_approx = np.array(list_approx)
    # replace 0s with 1s in the denominator to prevent errors
    vals_exact_denom = np.where(vals_exact==0,1,vals_exact)
    gap = np.mean(np.abs(vals_approx - vals_exact)/ vals_exact_denom)
    return gap


def approx_ratio(exact, heur):
    exact_p = np.array([exact])
    heur_p = np.array([heur])
    exact_denom = np.where(exact_p == 0, 1, exact_p)
    heur_denom = np.where(heur_p == 0, 1, heur_p)
    c = np.concatenate((exact_denom / heur_denom, heur_denom / exact_denom), axis=0)
    ratio = np.max(c, axis=0)
    mean_ratio = np.mean(ratio)
    return mean_ratio
