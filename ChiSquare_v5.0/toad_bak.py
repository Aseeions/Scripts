import numpy as np
cimport numpy as np
cimport cython
from .utils import fillna, to_ndarray
from .c_utils cimport c_min, c_sum, c_sum_axis_0, c_sum_axis_1


cpdef ChiMerge(feature, target, n_bins = None, min_samples = None, min_threshold = None, nan = -1, balance = True):
"""Chi-Merge
Args:
    feature (array-like): feature to be merged
    target (array-like): a array of target classes
    n_bins (int): n bins will be merged into
    min_samples (number): min sample in each group, if float, it will be the percentage of samples
    min_threshold (number): min threshold of chi-square
Returns:
    array: array of split points
"""

# set default break condition
if n_bins is None and min_samples is None and min_threshold is None:
    n_bins = DEFAULT_BINS

if min_samples and min_samples < 1:
    min_samples = len(feature) * min_samples

feature = fillna(feature, by = nan)
target = to_ndarray(target)


target_unique = np.unique(target)
feature_unique = np.unique(feature)
len_f = len(feature_unique)
len_t = len(target_unique)

cdef double [:,:] grouped = np.zeros((len_f, len_t), dtype=np.float)

for r in range(len_f):
    tmp = target[feature == feature_unique[r]]
    for c in range(len_t):
        grouped[r, c] = (tmp == target_unique[c]).sum()


cdef double [:,:] couple
cdef double [:] cols, rows, chi_list
# cdef long [:] min_ix, drop_ix
# cdef long[:] chi_ix
cdef double chi, chi_min, total, e
cdef int l, retain_ix, ix
cdef Py_ssize_t i, j, k, p

while(True):
    # break loop when reach n_bins
    if n_bins and len(grouped) <= n_bins:
        break

    # break loop if min samples of groups is greater than threshold
    if min_samples and c_min(c_sum_axis_1(grouped)) > min_samples:
        break

    # Calc chi square for each group
    l = len(grouped) - 1
    chi_list = np.zeros(l, dtype=np.float)
    chi_min = np.inf
    # chi_ix = []
    for i in range(l):
        chi = 0
        couple = grouped[i:i+2,:]
        total = c_sum(couple)
        cols = c_sum_axis_0(couple)
        rows = c_sum_axis_1(couple)

        for j in range(couple.shape[0]):
            for k in range(couple.shape[1]):
                e = rows[j] * cols[k] / total
                if e != 0:
                    chi += (couple[j, k] - e) ** 2 / e

        # balance weight of chi
        if balance:
            chi *= total

        chi_list[i] = chi

        if chi == chi_min:
            chi_ix.append(i)
            continue

        if chi < chi_min:
            chi_min = chi
            chi_ix = [i]


    # break loop when the minimun chi greater the threshold
    if min_threshold and chi_min > min_threshold:
        break

    # get indexes of the groups who has the minimun chi
    min_ix = np.array(chi_ix)
    # min_ix = np.where(chi_list == chi_min)[0]

    # get the indexes witch needs to drop
    drop_ix = min_ix + 1


    # combine groups by indexes
    retain_ix = min_ix[0]
    last_ix = retain_ix
    for ix in min_ix:
        # set a new group
        if ix - last_ix > 1:
            retain_ix = ix

        # combine all contiguous indexes into one group
        for p in range(grouped.shape[1]):
            grouped[retain_ix, p] = grouped[retain_ix, p] + grouped[ix + 1, p]

        last_ix = ix


    # drop binned groups
    grouped = np.delete(grouped, drop_ix, axis = 0)
    feature_unique = np.delete(feature_unique, drop_ix)


return feature_unique[1:]










