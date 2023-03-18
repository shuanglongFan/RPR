import numpy as np
import pandas as pd
import math
import cvxpy

def cv_group(k, label, seed):
    """
    generate the group of k-fold cross validation
    parameters:
        k: folds
        label: labels represented as 0 and 1
    return:
    [[obs of the 1st fold], [obs of the 2nd fold], ...]
    """

    np.random.seed(seed)
    temp = pd.DataFrame(np.array(label), columns=["label"])
    temp_class_one = temp.loc[temp.label == 0]
    temp_class_two = temp.loc[temp.label == 1]
    n_class_one = (list(range(1, k+1))*math.ceil((temp_class_one.shape[0]/k)))[:temp_class_one.shape[0]]
    n_class_two = (list(range(1, k+1))*math.ceil((temp_class_two.shape[0]/k)))[:temp_class_two.shape[0]]
    temp_class_one = temp_class_one.copy()
    temp_class_two = temp_class_two.copy()
    temp_class_one['k'] = list(np.random.choice(a=n_class_one, size=temp_class_one.shape[0], replace=None))
    temp_class_two['k'] = list(np.random.choice(a=n_class_two, size=temp_class_two.shape[0], replace=None))
    
    cv_list = list()
    for x in range(1, k+1):
        obs_class_one = list(temp_class_one.loc[temp_class_one.k == x].index)
        obs_class_two = list(temp_class_two.loc[temp_class_two.k == x].index)
        cv_list.append((obs_class_one + obs_class_two))
    return cv_list

def ho_group(train_num, label, seed):
    """
    generate the group of hold-out test
    parameters:
        train_num: size of the training set
        label: labels represented as 0 and 1
    return:
    [[obs of the training set], [obs of the testing set]]
    """
    
    np.random.seed(seed)
    sample_rate = train_num / len(label)
    temp = pd.DataFrame(label, columns=["label"])
    temp_class_one = temp.loc[temp.label == 0]
    temp_class_two = temp.loc[temp.label == 1]
    sample_num_one = math.ceil((temp_class_one.shape[0]) * sample_rate)
    sample_num_two = train_num - sample_num_one
    train_index_one = list(np.random.choice(a=temp_class_one.index, size=sample_num_one, replace=None))
    train_index_two = list(np.random.choice(a=temp_class_two.index, size=sample_num_two, replace=None))
    train_index = train_index_one + train_index_two
    test_index = list(set(temp.index) - set(train_index))
    ho_list = [train_index, test_index]
    return ho_list

def hankel_mat(n, l):
    """
    generate hankel matrix
    parameters:
        n: dim of n*n
        l: if i+j = l, mat(i,j) = 1 else = 0
    return: 
        n*n matrix
    """
    
    mat = np.zeros((n, n))
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i+j == l:
                mat[i-1, j-1] = 1
            else:
                mat[i-1, j-1] = 0
    return mat

def shape_poly(pre_prob, true_label, degree, lamb, verbose=True):
    """
    Shape-restricted polynomial regression
    parameters:
        pre_prob: pre-calibration probabilities
        true_label: labels represented as 0 and 1
        degree: 4-20
        lamb: the constant for l1 norm
    return:
    [(degree, lamb, s_inf, s_sup), [coef_a]]
    """
    
    # coefficient matrix
    pre_prob = np.array(pre_prob)
    true_label = np.array(true_label)
    s_mat_train = np.zeros((len(pre_prob), degree+1))
    s_mat_train[:, 0] = 1
    for col_num in range(1, degree+1):
        s_mat_train[:, col_num] = pre_prob ** col_num
    
    # decesion variable
    s_inf_index = np.argmin(pre_prob)
    s_sup_index = np.argmax(pre_prob)
    s_inf = np.min(pre_prob)  # the lower bound of the score
    s_sup = np.max(pre_prob)  # the upper bound of the score
    coef_a = cp.Variable(degree+1)
    
    # optimization goal
    loss_mse = cp.sum_squares(s_mat_train @ coef_a - true_label)
    obj = cp.Minimize(loss_mse)
    
    # constrains
    cons1 = [s_mat_train[s_inf_index, :] @ coef_a >= 0]
    cons2 = [s_mat_train[s_sup_index, :] @ coef_a <= 1]
    cons3 = [cp.sum(cp.abs(coef_a)) <= lamb]
    
    # derivative constrains
    if degree == 4:
        k1 = 2
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(2, 2) + hankel_mat(2, 1)) * U) + cp.sum((s_sup * hankel_mat(2, 2) - hankel_mat(2, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(2, 3) + hankel_mat(2, 2)) * U) + cp.sum((s_sup * hankel_mat(2, 3) - hankel_mat(2, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(2, 4) + hankel_mat(2, 3)) * U) + cp.sum((s_sup * hankel_mat(2, 4) - hankel_mat(2, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(2, 1) + hankel_mat(2, 4)) * U) + cp.sum((s_sup * hankel_mat(2, 1) - hankel_mat(2, 4)) * V)]
    elif degree == 6:
        k1 = 3
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(3, 2) + hankel_mat(3, 1)) * U) + cp.sum((s_sup * hankel_mat(3, 2) - hankel_mat(3, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(3, 3) + hankel_mat(3, 2)) * U) + cp.sum((s_sup * hankel_mat(3, 3) - hankel_mat(3, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(3, 4) + hankel_mat(3, 3)) * U) + cp.sum((s_sup * hankel_mat(3, 4) - hankel_mat(3, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(3, 5) + hankel_mat(3, 4)) * U) + cp.sum((s_sup * hankel_mat(3, 5) - hankel_mat(3, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(3, 6) + hankel_mat(3, 5)) * U) + cp.sum((s_sup * hankel_mat(3, 6) - hankel_mat(3, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(3, 1) + hankel_mat(3, 6)) * U) + cp.sum((s_sup * hankel_mat(3, 1) - hankel_mat(3, 6)) * V)]
    elif degree == 8:
        k1 = 4
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(4, 2) + hankel_mat(4, 1)) * U) + cp.sum((s_sup * hankel_mat(4, 2) - hankel_mat(4, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(4, 3) + hankel_mat(4, 2)) * U) + cp.sum((s_sup * hankel_mat(4, 3) - hankel_mat(4, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(4, 4) + hankel_mat(4, 3)) * U) + cp.sum((s_sup * hankel_mat(4, 4) - hankel_mat(4, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(4, 5) + hankel_mat(4, 4)) * U) + cp.sum((s_sup * hankel_mat(4, 5) - hankel_mat(4, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(4, 6) + hankel_mat(4, 5)) * U) + cp.sum((s_sup * hankel_mat(4, 6) - hankel_mat(4, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(4, 7) + hankel_mat(4, 6)) * U) + cp.sum((s_sup * hankel_mat(4, 7) - hankel_mat(4, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(4, 8) + hankel_mat(4, 7)) * U) + cp.sum((s_sup * hankel_mat(4, 8) - hankel_mat(4, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(4, 1) + hankel_mat(4, 8)) * U) + cp.sum((s_sup * hankel_mat(4, 1) - hankel_mat(4, 8)) * V)]
    elif degree == 10:
        k1 = 5
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(5, 2) + hankel_mat(5, 1)) * U) + cp.sum((s_sup * hankel_mat(5, 2) - hankel_mat(5, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(5, 3) + hankel_mat(5, 2)) * U) + cp.sum((s_sup * hankel_mat(5, 3) - hankel_mat(5, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(5, 4) + hankel_mat(5, 3)) * U) + cp.sum((s_sup * hankel_mat(5, 4) - hankel_mat(5, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(5, 5) + hankel_mat(5, 4)) * U) + cp.sum((s_sup * hankel_mat(5, 5) - hankel_mat(5, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(5, 6) + hankel_mat(5, 5)) * U) + cp.sum((s_sup * hankel_mat(5, 6) - hankel_mat(5, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(5, 7) + hankel_mat(5, 6)) * U) + cp.sum((s_sup * hankel_mat(5, 7) - hankel_mat(5, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(5, 8) + hankel_mat(5, 7)) * U) + cp.sum((s_sup * hankel_mat(5, 8) - hankel_mat(5, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(5, 9) + hankel_mat(5, 8)) * U) + cp.sum((s_sup * hankel_mat(5, 9) - hankel_mat(5, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(5, 10) + hankel_mat(5, 9)) * U) + cp.sum((s_sup * hankel_mat(5, 10) - hankel_mat(5, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(5, 1) + hankel_mat(5, 10)) * U) + cp.sum((s_sup * hankel_mat(5, 1) - hankel_mat(5, 10)) * V)]
    elif degree == 12:
        k1 = 6
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(6, 2) + hankel_mat(6, 1)) * U) + cp.sum((s_sup * hankel_mat(6, 2) - hankel_mat(6, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(6, 3) + hankel_mat(6, 2)) * U) + cp.sum((s_sup * hankel_mat(6, 3) - hankel_mat(6, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(6, 4) + hankel_mat(6, 3)) * U) + cp.sum((s_sup * hankel_mat(6, 4) - hankel_mat(6, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(6, 5) + hankel_mat(6, 4)) * U) + cp.sum((s_sup * hankel_mat(6, 5) - hankel_mat(6, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(6, 6) + hankel_mat(6, 5)) * U) + cp.sum((s_sup * hankel_mat(6, 6) - hankel_mat(6, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(6, 7) + hankel_mat(6, 6)) * U) + cp.sum((s_sup * hankel_mat(6, 7) - hankel_mat(6, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(6, 8) + hankel_mat(6, 7)) * U) + cp.sum((s_sup * hankel_mat(6, 8) - hankel_mat(6, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(6, 9) + hankel_mat(6, 8)) * U) + cp.sum((s_sup * hankel_mat(6, 9) - hankel_mat(6, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(6, 10) + hankel_mat(6, 9)) * U) + cp.sum((s_sup * hankel_mat(6, 10) - hankel_mat(6, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(6, 11) + hankel_mat(6, 10)) * U) + cp.sum((s_sup * hankel_mat(6, 11) - hankel_mat(6, 10)) * V),
                         11 * coef_a[11] == cp.sum((-s_inf * hankel_mat(6, 12) + hankel_mat(6, 11)) * U) + cp.sum((s_sup * hankel_mat(6, 12) - hankel_mat(6, 11)) * V),
                         12 * coef_a[12] == cp.sum((-s_inf * hankel_mat(6, 1) + hankel_mat(6, 12)) * U) + cp.sum((s_sup * hankel_mat(6, 1) - hankel_mat(6, 12)) * V)]
    elif degree == 14:
        k1 = 7
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(7, 2) + hankel_mat(7, 1)) * U) + cp.sum((s_sup * hankel_mat(7, 2) - hankel_mat(7, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(7, 3) + hankel_mat(7, 2)) * U) + cp.sum((s_sup * hankel_mat(7, 3) - hankel_mat(7, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(7, 4) + hankel_mat(7, 3)) * U) + cp.sum((s_sup * hankel_mat(7, 4) - hankel_mat(7, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(7, 5) + hankel_mat(7, 4)) * U) + cp.sum((s_sup * hankel_mat(7, 5) - hankel_mat(7, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(7, 6) + hankel_mat(7, 5)) * U) + cp.sum((s_sup * hankel_mat(7, 6) - hankel_mat(7, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(7, 7) + hankel_mat(7, 6)) * U) + cp.sum((s_sup * hankel_mat(7, 7) - hankel_mat(7, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(7, 8) + hankel_mat(7, 7)) * U) + cp.sum((s_sup * hankel_mat(7, 8) - hankel_mat(7, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(7, 9) + hankel_mat(7, 8)) * U) + cp.sum((s_sup * hankel_mat(7, 9) - hankel_mat(7, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(7, 10) + hankel_mat(7, 9)) * U) + cp.sum((s_sup * hankel_mat(7, 10) - hankel_mat(7, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(7, 11) + hankel_mat(7, 10)) * U) + cp.sum((s_sup * hankel_mat(7, 11) - hankel_mat(7, 10)) * V),
                         11 * coef_a[11] == cp.sum((-s_inf * hankel_mat(7, 12) + hankel_mat(7, 11)) * U) + cp.sum((s_sup * hankel_mat(7, 12) - hankel_mat(7, 11)) * V),
                         12 * coef_a[12] == cp.sum((-s_inf * hankel_mat(7, 13) + hankel_mat(7, 12)) * U) + cp.sum((s_sup * hankel_mat(7, 13) - hankel_mat(7, 12)) * V),
                         13 * coef_a[13] == cp.sum((-s_inf * hankel_mat(7, 14) + hankel_mat(7, 13)) * U) + cp.sum((s_sup * hankel_mat(7, 14) - hankel_mat(7, 13)) * V),
                         14 * coef_a[14] == cp.sum((-s_inf * hankel_mat(7, 1) + hankel_mat(7, 14)) * U) + cp.sum((s_sup * hankel_mat(7, 1) - hankel_mat(7, 14)) * V)]
    elif degree == 16:
        k1 = 8
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(8, 2) + hankel_mat(8, 1)) * U) + cp.sum((s_sup * hankel_mat(8, 2) - hankel_mat(8, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(8, 3) + hankel_mat(8, 2)) * U) + cp.sum((s_sup * hankel_mat(8, 3) - hankel_mat(8, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(8, 4) + hankel_mat(8, 3)) * U) + cp.sum((s_sup * hankel_mat(8, 4) - hankel_mat(8, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(8, 5) + hankel_mat(8, 4)) * U) + cp.sum((s_sup * hankel_mat(8, 5) - hankel_mat(8, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(8, 6) + hankel_mat(8, 5)) * U) + cp.sum((s_sup * hankel_mat(8, 6) - hankel_mat(8, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(8, 7) + hankel_mat(8, 6)) * U) + cp.sum((s_sup * hankel_mat(8, 7) - hankel_mat(8, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(8, 8) + hankel_mat(8, 7)) * U) + cp.sum((s_sup * hankel_mat(8, 8) - hankel_mat(8, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(8, 9) + hankel_mat(8, 8)) * U) + cp.sum((s_sup * hankel_mat(8, 9) - hankel_mat(8, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(8, 10) + hankel_mat(8, 9)) * U) + cp.sum((s_sup * hankel_mat(8, 10) - hankel_mat(8, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(8, 11) + hankel_mat(8, 10)) * U) + cp.sum((s_sup * hankel_mat(8, 11) - hankel_mat(8, 10)) * V),
                         11 * coef_a[11] == cp.sum((-s_inf * hankel_mat(8, 12) + hankel_mat(8, 11)) * U) + cp.sum((s_sup * hankel_mat(8, 12) - hankel_mat(8, 11)) * V),
                         12 * coef_a[12] == cp.sum((-s_inf * hankel_mat(8, 13) + hankel_mat(8, 12)) * U) + cp.sum((s_sup * hankel_mat(8, 13) - hankel_mat(8, 12)) * V),
                         13 * coef_a[13] == cp.sum((-s_inf * hankel_mat(8, 14) + hankel_mat(8, 13)) * U) + cp.sum((s_sup * hankel_mat(8, 14) - hankel_mat(8, 13)) * V),
                         14 * coef_a[14] == cp.sum((-s_inf * hankel_mat(8, 15) + hankel_mat(8, 14)) * U) + cp.sum((s_sup * hankel_mat(8, 15) - hankel_mat(8, 14)) * V),
                         15 * coef_a[15] == cp.sum((-s_inf * hankel_mat(8, 16) + hankel_mat(8, 15)) * U) + cp.sum((s_sup * hankel_mat(8, 16) - hankel_mat(8, 15)) * V),
                         16 * coef_a[16] == cp.sum((-s_inf * hankel_mat(8, 1) + hankel_mat(8, 16)) * U) + cp.sum((s_sup * hankel_mat(8, 1) - hankel_mat(8, 16)) * V)]
    elif degree == 18:
        k1 = 9
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(9, 2) + hankel_mat(9, 1)) * U) + cp.sum((s_sup * hankel_mat(9, 2) - hankel_mat(9, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(9, 3) + hankel_mat(9, 2)) * U) + cp.sum((s_sup * hankel_mat(9, 3) - hankel_mat(9, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(9, 4) + hankel_mat(9, 3)) * U) + cp.sum((s_sup * hankel_mat(9, 4) - hankel_mat(9, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(9, 5) + hankel_mat(9, 4)) * U) + cp.sum((s_sup * hankel_mat(9, 5) - hankel_mat(9, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(9, 6) + hankel_mat(9, 5)) * U) + cp.sum((s_sup * hankel_mat(9, 6) - hankel_mat(9, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(9, 7) + hankel_mat(9, 6)) * U) + cp.sum((s_sup * hankel_mat(9, 7) - hankel_mat(9, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(9, 8) + hankel_mat(9, 7)) * U) + cp.sum((s_sup * hankel_mat(9, 8) - hankel_mat(9, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(9, 9) + hankel_mat(9, 8)) * U) + cp.sum((s_sup * hankel_mat(9, 9) - hankel_mat(9, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(9, 10) + hankel_mat(9, 9)) * U) + cp.sum((s_sup * hankel_mat(9, 10) - hankel_mat(9, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(9, 11) + hankel_mat(9, 10)) * U) + cp.sum((s_sup * hankel_mat(9, 11) - hankel_mat(9, 10)) * V),
                         11 * coef_a[11] == cp.sum((-s_inf * hankel_mat(9, 12) + hankel_mat(9, 11)) * U) + cp.sum((s_sup * hankel_mat(9, 12) - hankel_mat(9, 11)) * V),
                         12 * coef_a[12] == cp.sum((-s_inf * hankel_mat(9, 13) + hankel_mat(9, 12)) * U) + cp.sum((s_sup * hankel_mat(9, 13) - hankel_mat(9, 12)) * V),
                         13 * coef_a[13] == cp.sum((-s_inf * hankel_mat(9, 14) + hankel_mat(9, 13)) * U) + cp.sum((s_sup * hankel_mat(9, 14) - hankel_mat(9, 13)) * V),
                         14 * coef_a[14] == cp.sum((-s_inf * hankel_mat(9, 15) + hankel_mat(9, 14)) * U) + cp.sum((s_sup * hankel_mat(9, 15) - hankel_mat(9, 14)) * V),
                         15 * coef_a[15] == cp.sum((-s_inf * hankel_mat(9, 16) + hankel_mat(9, 15)) * U) + cp.sum((s_sup * hankel_mat(9, 16) - hankel_mat(9, 15)) * V),
                         16 * coef_a[16] == cp.sum((-s_inf * hankel_mat(9, 17) + hankel_mat(9, 16)) * U) + cp.sum((s_sup * hankel_mat(9, 17) - hankel_mat(9, 16)) * V),
                         17 * coef_a[17] == cp.sum((-s_inf * hankel_mat(9, 18) + hankel_mat(9, 17)) * U) + cp.sum((s_sup * hankel_mat(9, 18) - hankel_mat(9, 17)) * V),
                         18 * coef_a[18] == cp.sum((-s_inf * hankel_mat(9, 1) + hankel_mat(9, 18)) * U) + cp.sum((s_sup * hankel_mat(9, 1) - hankel_mat(9, 18)) * V)]
    elif degree == 20:
        k1 = 10
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1, k1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum((-s_inf * hankel_mat(10, 2) + hankel_mat(10, 1)) * U) + cp.sum((s_sup * hankel_mat(10, 2) - hankel_mat(10, 1)) * V),
                         2 * coef_a[2] == cp.sum((-s_inf * hankel_mat(10, 3) + hankel_mat(10, 2)) * U) + cp.sum((s_sup * hankel_mat(10, 3) - hankel_mat(10, 2)) * V),
                         3 * coef_a[3] == cp.sum((-s_inf * hankel_mat(10, 4) + hankel_mat(10, 3)) * U) + cp.sum((s_sup * hankel_mat(10, 4) - hankel_mat(10, 3)) * V),
                         4 * coef_a[4] == cp.sum((-s_inf * hankel_mat(10, 5) + hankel_mat(10, 4)) * U) + cp.sum((s_sup * hankel_mat(10, 5) - hankel_mat(10, 4)) * V),
                         5 * coef_a[5] == cp.sum((-s_inf * hankel_mat(10, 6) + hankel_mat(10, 5)) * U) + cp.sum((s_sup * hankel_mat(10, 6) - hankel_mat(10, 5)) * V),
                         6 * coef_a[6] == cp.sum((-s_inf * hankel_mat(10, 7) + hankel_mat(10, 6)) * U) + cp.sum((s_sup * hankel_mat(10, 7) - hankel_mat(10, 6)) * V),
                         7 * coef_a[7] == cp.sum((-s_inf * hankel_mat(10, 8) + hankel_mat(10, 7)) * U) + cp.sum((s_sup * hankel_mat(10, 8) - hankel_mat(10, 7)) * V),
                         8 * coef_a[8] == cp.sum((-s_inf * hankel_mat(10, 9) + hankel_mat(10, 8)) * U) + cp.sum((s_sup * hankel_mat(10, 9) - hankel_mat(10, 8)) * V),
                         9 * coef_a[9] == cp.sum((-s_inf * hankel_mat(10, 10) + hankel_mat(10, 9)) * U) + cp.sum((s_sup * hankel_mat(10, 10) - hankel_mat(10, 9)) * V),
                         10 * coef_a[10] == cp.sum((-s_inf * hankel_mat(10, 11) + hankel_mat(10, 10)) * U) + cp.sum((s_sup * hankel_mat(10, 11) - hankel_mat(10, 10)) * V),
                         11 * coef_a[11] == cp.sum((-s_inf * hankel_mat(10, 12) + hankel_mat(10, 11)) * U) + cp.sum((s_sup * hankel_mat(10, 12) - hankel_mat(10, 11)) * V),
                         12 * coef_a[12] == cp.sum((-s_inf * hankel_mat(10, 13) + hankel_mat(10, 12)) * U) + cp.sum((s_sup * hankel_mat(10, 13) - hankel_mat(10, 12)) * V),
                         13 * coef_a[13] == cp.sum((-s_inf * hankel_mat(10, 14) + hankel_mat(10, 13)) * U) + cp.sum((s_sup * hankel_mat(10, 14) - hankel_mat(10, 13)) * V),
                         14 * coef_a[14] == cp.sum((-s_inf * hankel_mat(10, 15) + hankel_mat(10, 14)) * U) + cp.sum((s_sup * hankel_mat(10, 15) - hankel_mat(10, 14)) * V),
                         15 * coef_a[15] == cp.sum((-s_inf * hankel_mat(10, 16) + hankel_mat(10, 15)) * U) + cp.sum((s_sup * hankel_mat(10, 16) - hankel_mat(10, 15)) * V),
                         16 * coef_a[16] == cp.sum((-s_inf * hankel_mat(10, 17) + hankel_mat(10, 16)) * U) + cp.sum((s_sup * hankel_mat(10, 17) - hankel_mat(10, 16)) * V),
                         17 * coef_a[17] == cp.sum((-s_inf * hankel_mat(10, 18) + hankel_mat(10, 17)) * U) + cp.sum((s_sup * hankel_mat(10, 18) - hankel_mat(10, 17)) * V),
                         18 * coef_a[18] == cp.sum((-s_inf * hankel_mat(10, 19) + hankel_mat(10, 18)) * U) + cp.sum((s_sup * hankel_mat(10, 19) - hankel_mat(10, 18)) * V),
                         19 * coef_a[19] == cp.sum((-s_inf * hankel_mat(10, 20) + hankel_mat(10, 19)) * U) + cp.sum((s_sup * hankel_mat(10, 20) - hankel_mat(10, 19)) * V),
                         20 * coef_a[20] == cp.sum((-s_inf * hankel_mat(10, 1) + hankel_mat(10, 20)) * U) + cp.sum((s_sup * hankel_mat(10, 1) - hankel_mat(10, 20)) * V)]
    elif degree == 5:
        k1 = 3
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(3, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(2, 2) + (s_inf + s_sup) * hankel_mat(2, 1) - hankel_mat(2, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(3, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(2, 3) + (s_inf + s_sup) * hankel_mat(2, 2) - hankel_mat(2, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(3, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(2, 4) + (s_inf + s_sup) * hankel_mat(2, 3) - hankel_mat(2, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(3, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(2, 1) + (s_inf + s_sup) * hankel_mat(2, 4) - hankel_mat(2, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(3, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(2, 1) + (s_inf + s_sup) * hankel_mat(2, 1) - hankel_mat(2, 4)) * V)]
    elif degree == 7:
        k1 = 4
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(4, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 2) + (s_inf + s_sup) * hankel_mat(3, 1) - hankel_mat(3, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(4, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 3) + (s_inf + s_sup) * hankel_mat(3, 2) - hankel_mat(3, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(4, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 4) + (s_inf + s_sup) * hankel_mat(3, 3) - hankel_mat(3, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(4, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 5) + (s_inf + s_sup) * hankel_mat(3, 4) - hankel_mat(3, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(4, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 6) + (s_inf + s_sup) * hankel_mat(3, 5) - hankel_mat(3, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(4, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 1) + (s_inf + s_sup) * hankel_mat(3, 6) - hankel_mat(3, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(4, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(3, 1) + (s_inf + s_sup) * hankel_mat(3, 1) - hankel_mat(3, 6)) * V)]
    elif degree == 9:
        k1 = 5
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(5, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 2) + (s_inf + s_sup) * hankel_mat(4, 1) - hankel_mat(4, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(5, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 3) + (s_inf + s_sup) * hankel_mat(4, 2) - hankel_mat(4, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(5, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 4) + (s_inf + s_sup) * hankel_mat(4, 3) - hankel_mat(4, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(5, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 5) + (s_inf + s_sup) * hankel_mat(4, 4) - hankel_mat(4, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(5, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 6) + (s_inf + s_sup) * hankel_mat(4, 5) - hankel_mat(4, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(5, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 7) + (s_inf + s_sup) * hankel_mat(4, 6) - hankel_mat(4, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(5, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 8) + (s_inf + s_sup) * hankel_mat(4, 7) - hankel_mat(4, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(5, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 1) + (s_inf + s_sup) * hankel_mat(4, 8) - hankel_mat(4, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(5, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(4, 1) + (s_inf + s_sup) * hankel_mat(4, 1) - hankel_mat(4, 8)) * V)]
    elif degree == 11:
        k1 = 6
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(6, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 2) + (s_inf + s_sup) * hankel_mat(5, 1) - hankel_mat(5, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(6, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 3) + (s_inf + s_sup) * hankel_mat(5, 2) - hankel_mat(5, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(6, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 4) + (s_inf + s_sup) * hankel_mat(5, 3) - hankel_mat(5, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(6, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 5) + (s_inf + s_sup) * hankel_mat(5, 4) - hankel_mat(5, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(6, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 6) + (s_inf + s_sup) * hankel_mat(5, 5) - hankel_mat(5, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(6, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 7) + (s_inf + s_sup) * hankel_mat(5, 6) - hankel_mat(5, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(6, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 8) + (s_inf + s_sup) * hankel_mat(5, 7) - hankel_mat(5, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(6, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 9) + (s_inf + s_sup) * hankel_mat(5, 8) - hankel_mat(5, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(6, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 10) + (s_inf + s_sup) * hankel_mat(5, 9) - hankel_mat(5, 8)) * V),
                         10 * coef_a[10] == cp.sum(hankel_mat(6, 11) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 1) + (s_inf + s_sup) * hankel_mat(5, 10) - hankel_mat(5, 9)) * V),
                         11 * coef_a[11] == cp.sum(hankel_mat(6, 12) * U) + cp.sum((-s_inf * s_sup * hankel_mat(5, 1) + (s_inf + s_sup) * hankel_mat(5, 1) - hankel_mat(5, 10)) * V)]
    elif degree == 13:
        k1 = 7
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(7, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 2) + (s_inf + s_sup) * hankel_mat(6, 1) - hankel_mat(6, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(7, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 3) + (s_inf + s_sup) * hankel_mat(6, 2) - hankel_mat(6, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(7, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 4) + (s_inf + s_sup) * hankel_mat(6, 3) - hankel_mat(6, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(7, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 5) + (s_inf + s_sup) * hankel_mat(6, 4) - hankel_mat(6, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(7, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 6) + (s_inf + s_sup) * hankel_mat(6, 5) - hankel_mat(6, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(7, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 7) + (s_inf + s_sup) * hankel_mat(6, 6) - hankel_mat(6, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(7, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 8) + (s_inf + s_sup) * hankel_mat(6, 7) - hankel_mat(6, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(7, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 9) + (s_inf + s_sup) * hankel_mat(6, 8) - hankel_mat(6, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(7, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 10) + (s_inf + s_sup) * hankel_mat(6, 9) - hankel_mat(6, 8)) * V),
                         10 * coef_a[10] == cp.sum(hankel_mat(7, 11) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 11) + (s_inf + s_sup) * hankel_mat(6, 10) - hankel_mat(6, 9)) * V),
                         11 * coef_a[11] == cp.sum(hankel_mat(7, 12) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 12) + (s_inf + s_sup) * hankel_mat(6, 11) - hankel_mat(6, 10)) * V),
                         12 * coef_a[12] == cp.sum(hankel_mat(7, 13) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 1) + (s_inf + s_sup) * hankel_mat(6, 12) - hankel_mat(6, 11)) * V),
                         13 * coef_a[13] == cp.sum(hankel_mat(7, 14) * U) + cp.sum((-s_inf * s_sup * hankel_mat(6, 1) + (s_inf + s_sup) * hankel_mat(6, 1) - hankel_mat(6, 12)) * V)]
    elif degree == 15:
        k1 = 8
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(8, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 2) + (s_inf + s_sup) * hankel_mat(7, 1) - hankel_mat(7, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(8, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 3) + (s_inf + s_sup) * hankel_mat(7, 2) - hankel_mat(7, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(8, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 4) + (s_inf + s_sup) * hankel_mat(7, 3) - hankel_mat(7, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(8, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 5) + (s_inf + s_sup) * hankel_mat(7, 4) - hankel_mat(7, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(8, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 6) + (s_inf + s_sup) * hankel_mat(7, 5) - hankel_mat(7, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(8, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 7) + (s_inf + s_sup) * hankel_mat(7, 6) - hankel_mat(7, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(8, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 8) + (s_inf + s_sup) * hankel_mat(7, 7) - hankel_mat(7, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(8, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 9) + (s_inf + s_sup) * hankel_mat(7, 8) - hankel_mat(7, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(8, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 10) + (s_inf + s_sup) * hankel_mat(7, 9) - hankel_mat(7, 8)) * V),
                         10 * coef_a[10] == cp.sum(hankel_mat(8, 11) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 11) + (s_inf + s_sup) * hankel_mat(7, 10) - hankel_mat(7, 9)) * V),
                         11 * coef_a[11] == cp.sum(hankel_mat(8, 12) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 12) + (s_inf + s_sup) * hankel_mat(7, 11) - hankel_mat(7, 10)) * V),
                         12 * coef_a[12] == cp.sum(hankel_mat(8, 13) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 13) + (s_inf + s_sup) * hankel_mat(7, 12) - hankel_mat(7, 11)) * V),
                         13 * coef_a[13] == cp.sum(hankel_mat(8, 14) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 14) + (s_inf + s_sup) * hankel_mat(7, 13) - hankel_mat(7, 12)) * V),
                         14 * coef_a[14] == cp.sum(hankel_mat(8, 15) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 1) + (s_inf + s_sup) * hankel_mat(7, 14) - hankel_mat(7, 13)) * V),
                         15 * coef_a[15] == cp.sum(hankel_mat(8, 16) * U) + cp.sum((-s_inf * s_sup * hankel_mat(7, 1) + (s_inf + s_sup) * hankel_mat(7, 1) - hankel_mat(7, 14)) * V)]
    elif degree == 17:
        k1 = 9
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(9, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 2) + (s_inf + s_sup) * hankel_mat(8, 1) - hankel_mat(8, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(9, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 3) + (s_inf + s_sup) * hankel_mat(8, 2) - hankel_mat(8, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(9, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 4) + (s_inf + s_sup) * hankel_mat(8, 3) - hankel_mat(8, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(9, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 5) + (s_inf + s_sup) * hankel_mat(8, 4) - hankel_mat(8, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(9, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 6) + (s_inf + s_sup) * hankel_mat(8, 5) - hankel_mat(8, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(9, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 7) + (s_inf + s_sup) * hankel_mat(8, 6) - hankel_mat(8, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(9, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 8) + (s_inf + s_sup) * hankel_mat(8, 7) - hankel_mat(8, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(9, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 9) + (s_inf + s_sup) * hankel_mat(8, 8) - hankel_mat(8, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(9, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 10) + (s_inf + s_sup) * hankel_mat(8, 9) - hankel_mat(8, 8)) * V),
                         10 * coef_a[10] == cp.sum(hankel_mat(9, 11) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 11) + (s_inf + s_sup) * hankel_mat(8, 10) - hankel_mat(8, 9)) * V),
                         11 * coef_a[11] == cp.sum(hankel_mat(9, 12) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 12) + (s_inf + s_sup) * hankel_mat(8, 11) - hankel_mat(8, 10)) * V),
                         12 * coef_a[12] == cp.sum(hankel_mat(9, 13) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 13) + (s_inf + s_sup) * hankel_mat(8, 12) - hankel_mat(8, 11)) * V),
                         13 * coef_a[13] == cp.sum(hankel_mat(9, 14) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 14) + (s_inf + s_sup) * hankel_mat(8, 13) - hankel_mat(8, 12)) * V),
                         14 * coef_a[14] == cp.sum(hankel_mat(9, 15) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 15) + (s_inf + s_sup) * hankel_mat(8, 14) - hankel_mat(8, 13)) * V),
                         15 * coef_a[15] == cp.sum(hankel_mat(9, 16) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 16) + (s_inf + s_sup) * hankel_mat(8, 15) - hankel_mat(8, 14)) * V),
                         16 * coef_a[16] == cp.sum(hankel_mat(9, 17) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 1) + (s_inf + s_sup) * hankel_mat(8, 16) - hankel_mat(8, 15)) * V),
                         17 * coef_a[17] == cp.sum(hankel_mat(9, 18) * U) + cp.sum((-s_inf * s_sup * hankel_mat(8, 1) + (s_inf + s_sup) * hankel_mat(8, 1) - hankel_mat(8, 16)) * V)]
    elif degree == 19:
        k1 = 10
        U, V = cp.Variable((k1, k1), PSD=True), cp.Variable((k1-1, k1-1), PSD=True)
        cons_monotone = [1 * coef_a[1] == cp.sum(hankel_mat(10, 2) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 2) + (s_inf + s_sup) * hankel_mat(9, 1) - hankel_mat(9, 1)) * V),
                         2 * coef_a[2] == cp.sum(hankel_mat(10, 3) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 3) + (s_inf + s_sup) * hankel_mat(9, 2) - hankel_mat(9, 1)) * V),
                         3 * coef_a[3] == cp.sum(hankel_mat(10, 4) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 4) + (s_inf + s_sup) * hankel_mat(9, 3) - hankel_mat(9, 2)) * V),
                         4 * coef_a[4] == cp.sum(hankel_mat(10, 5) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 5) + (s_inf + s_sup) * hankel_mat(9, 4) - hankel_mat(9, 3)) * V),
                         5 * coef_a[5] == cp.sum(hankel_mat(10, 6) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 6) + (s_inf + s_sup) * hankel_mat(9, 5) - hankel_mat(9, 4)) * V),
                         6 * coef_a[6] == cp.sum(hankel_mat(10, 7) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 7) + (s_inf + s_sup) * hankel_mat(9, 6) - hankel_mat(9, 5)) * V),
                         7 * coef_a[7] == cp.sum(hankel_mat(10, 8) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 8) + (s_inf + s_sup) * hankel_mat(9, 7) - hankel_mat(9, 6)) * V),
                         8 * coef_a[8] == cp.sum(hankel_mat(10, 9) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 9) + (s_inf + s_sup) * hankel_mat(9, 8) - hankel_mat(9, 7)) * V),
                         9 * coef_a[9] == cp.sum(hankel_mat(10, 10) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 10) + (s_inf + s_sup) * hankel_mat(9, 9) - hankel_mat(9, 8)) * V),
                         10 * coef_a[10] == cp.sum(hankel_mat(10, 11) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 11) + (s_inf + s_sup) * hankel_mat(9, 10) - hankel_mat(9, 9)) * V),
                         11 * coef_a[11] == cp.sum(hankel_mat(10, 12) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 12) + (s_inf + s_sup) * hankel_mat(9, 11) - hankel_mat(9, 10)) * V),
                         12 * coef_a[12] == cp.sum(hankel_mat(10, 13) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 13) + (s_inf + s_sup) * hankel_mat(9, 12) - hankel_mat(9, 11)) * V),
                         13 * coef_a[13] == cp.sum(hankel_mat(10, 14) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 14) + (s_inf + s_sup) * hankel_mat(9, 13) - hankel_mat(9, 12)) * V),
                         14 * coef_a[14] == cp.sum(hankel_mat(10, 15) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 15) + (s_inf + s_sup) * hankel_mat(9, 14) - hankel_mat(9, 13)) * V),
                         15 * coef_a[15] == cp.sum(hankel_mat(10, 16) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 16) + (s_inf + s_sup) * hankel_mat(9, 15) - hankel_mat(9, 14)) * V),
                         16 * coef_a[16] == cp.sum(hankel_mat(10, 17) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 17) + (s_inf + s_sup) * hankel_mat(9, 16) - hankel_mat(9, 15)) * V),
                         17 * coef_a[17] == cp.sum(hankel_mat(10, 18) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 18) + (s_inf + s_sup) * hankel_mat(9, 17) - hankel_mat(9, 16)) * V),
                         18 * coef_a[18] == cp.sum(hankel_mat(10, 19) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 1) + (s_inf + s_sup) * hankel_mat(9, 18) - hankel_mat(9, 17)) * V),
                         19 * coef_a[19] == cp.sum(hankel_mat(10, 20) * U) + cp.sum((-s_inf * s_sup * hankel_mat(9, 1) + (s_inf + s_sup) * hankel_mat(9, 1) - hankel_mat(9, 18)) * V)]
    elif degree not in range(4, 21):
        print("Degree out of the bounds")
        return
    
    cons_all = cons1 + cons2 + cons3 + cons_monotone
    problem = cp.Problem(obj, cons_all)
    problem.solve(verbose=verbose, max_iters=5000)
    a_star = coef_a.value
    return_info = (degree, lamb, s_inf, s_sup)
    return [return_info, a_star]
    
    print("The problem status is: %s" % problem.status)
    print("degree = %d, lambda = %.4f, s_inf = %.4f, s_sup = %.4f, loss_se = %.4f" % (degree, lamb, s_inf, s_sup, obj.value))
    print("The sum of the absolute values of a is：%.4f" % sum(abs(a_star)))
    print("**" * 40)
    print("")
  
def measure_cali(true_label, pre_prob, group=10):
    """
    measure calibration performance
    parameters:
        true_label: coded with 0 and 1
        pre_prob: probabilities
        group: bin num
    return:
    (MSE, ECE, MCE)
    """
    
    data = pd.DataFrame({"pre_prob": np.array(pre_prob), "true_label": np.array(true_label)})
    data = data.sort_values("pre_prob", ascending=True, ignore_index=True)  # must ignore index, or can not slice
    mse = sum((data["pre_prob"] - data["true_label"]) ** 2) / data.shape[0]
    
    # compute ECE and MCE
    group_error = list()
    group_size = round(data.shape[0] / group)
    
    i = 0
    r = 1
    while r <= group:
        if r == group:
            data_temp = data.loc[i:, :]
            group_error.append(abs(data_temp["true_label"].mean() - data_temp["pre_prob"].mean()))
            break
        data_temp = data.loc[i:i+group_size-1, :]
        group_error.append(abs(data_temp["true_label"].mean() - data_temp["pre_prob"].mean()))
        i += group_size
        r += 1
    ece, mce = np.mean(group_error), np.max(group_error)
    return mse, ece, mce

