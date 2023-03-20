import pandas as pd
import numpy as np
import main_funs as funs
import datetime
import os

def rpr_cali(ho_num, k_num, bin_num, data_dire, lower_degree=4, upper_degree=15, lower_lambda=-2, upper_lambda=2):
    """
    parameters:
        ho_num: the number of hold-out test
        k_num: k-fold cross-validation
        bin_num: used for calibration evaluation in validation data
	data_dire: the directory must contain true_label_train.csv pre_prob_vali.csv pre_prob_test.csv
        degree: polynomial degree
        lambda: the constant for l1 norm
    """
    
    os.chdir(data_dire)
    para_degree = list(range(lower_degree, upper_degree))
    para_lamb = [5 ** x for x in range(lower_lambda, upper_lambda, 1)]

    # loading data
    data_label = pd.read_csv("true_label_train.csv")  # each column corresponds to the whole training data of one hold-out test
    data_prob = pd.read_csv("pre_prob_vali.csv")  # each column corresponds to the whole validation data of one hold-out test

    # setting the combination of hyper-parameters
    para_grid = pd.DataFrame(columns=["degree", "lamb"])
    para_grid["degree"] = [x for x in para_degree for i in range(len(para_lamb))]
    para_grid["lamb"] = para_lamb * len(para_degree)

    # performing cross-validation to find optimal hyper-parameters
    measure_result = np.zeros((para_grid.shape[0], 3, ho_num))  # MSE, ECE, MCE

    # the union of the calibrated probabilities in validation data
    prob_vali_rpr = np.zeros((para_grid.shape[0], data_label.shape[0], ho_num))

    for col in range(0, para_grid.shape[0]):
        degree, lamb = para_grid.loc[col, "degree"], para_grid.loc[col, "lamb"]

        for seed_num in range(0, ho_num):
            data = pd.DataFrame(columns=["prob", "label"])
            data["prob"], data["label"] = data_prob.iloc[:, seed_num].copy(), data_label.iloc[:, seed_num].copy()
            cv_list = funs.cv_group(k=k_num, label=data["label"], seed=seed_num)
            measure_one_cv = np.zeros((3, k_num))
            prob_vali_rpr_temp = pd.DataFrame()  # extracting the union of calibrated probabilities in validation data

            for k in range(0, k_num):
                print("\t\t\t try paras: %d，hold-out num %d，validation num %d" % (col, seed_num, k))
                index_all = list(range(0, data.shape[0]))
                index_train, index_test = list(set(index_all) - set(cv_list[k])), cv_list[k]
                data_train, data_test = data.iloc[index_train, :].copy(), data.iloc[index_test, :].copy()  # avoiding chained index
                s_inf, s_sup = min(data_train["prob"]), max(data_train["prob"])
                data_test_temp = data_test.copy()
                data_test_temp.loc[data_test_temp["prob"] < s_inf, "prob"] = s_inf  # minimum score of training data
                data_test_temp.loc[data_test_temp["prob"] > s_sup, "prob"] = s_sup  # maximum score of training data
                s_mat_train, s_mat_test = np.zeros((len(index_train), degree + 1)), np.zeros((len(index_test), degree + 1))
                s_mat_train[:, 0], s_mat_test[:, 0] = 1, 1

                # constructing coefficient matrix of polynomial
                for i in range(1, degree+1):
                    s_mat_train[:, i] = data_train["prob"] ** i
                    s_mat_test[:, i] = data_test_temp["prob"] ** i

                poly_result = funs.shape_poly(pre_prob=data_train["prob"], true_label=data_train["label"], degree=degree, lamb=lamb, verbose=True)
                data_test["p_cali"] = s_mat_test @ poly_result[1]
                measure_one_cv[:, k] = funs.measure_cali(true_label=data_test["label"], pre_prob=data_test["p_cali"], group=bin_num)
                data_test["obs"] = index_test
                prob_vali_rpr_temp = pd.concat([prob_vali_rpr_temp, data_test], axis=0, ignore_index=True)

            measure_result[col, :, seed_num] = np.mean(measure_one_cv, axis=1)
            prob_vali_rpr_temp = prob_vali_rpr_temp.sort_values(["obs"])
            prob_vali_rpr[col, :, seed_num] = prob_vali_rpr_temp["p_cali"]

        t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("****" * 15)
        print("****\t\tparas：degree=%d, lambda=%.6f\t\t\t****\n****\t\t  %s, Done\t\t\t\t****" % (degree, lamb, t))
        print("****" * 15)
        print("")

    # determine optimal hyper-parameters based on MCE, and fitting the polynomial with the whole training data
    print("==" * 25)
    print("\tfitting the polynomial regression with the optimal hyper-parameters and whole training data")
    print("")
    mce_minimum_index = np.argmin(measure_result[:, 2, :], axis=0)
    vali_rpr = np.zeros((data_label.shape[0], ho_num))

    # the union of the calibrated probabilities of the validation data in each hold-out test
    for seed_num in range(0, ho_num):
        vali_rpr[:, seed_num] = prob_vali_rpr[mce_minimum_index[seed_num], :, seed_num]

    # loading prediction probabilities on the testing data and performing calibration
    pre_prob_test = pd.read_csv("pre_prob_test.csv")
    rpr_prob_test = np.zeros((pre_prob_test.shape[0], ho_num))

    for seed_num in range(0, ho_num):
        data = pd.DataFrame(columns=["prob", "label"])
        data["prob"], data["label"] = data_prob.iloc[:, seed_num].copy(), data_label.iloc[:, seed_num].copy()  # whole training data
        mce_minimum_index_temp = mce_minimum_index[seed_num]
        degree_star, lamb_star = para_grid["degree"][mce_minimum_index_temp], para_grid["lamb"][mce_minimum_index_temp]
        poly_star = funs.shape_poly(pre_prob=data["prob"], true_label=data["label"], degree=degree_star, lamb=lamb_star, verbose=False)
        s_inf, s_sup = poly_star[0][2], poly_star[0][3]
        
        # constructing coefficient matrix of polynomial and computing calibrated probabilities by RPR
        pre_prob_test_temp = pre_prob_test.iloc[:, seed_num].copy()
        pre_prob_test_temp[pre_prob_test_temp < s_inf] = s_inf
        pre_prob_test_temp[pre_prob_test_temp > s_sup] = s_sup
        s_mat = np.zeros((pre_prob_test_temp.shape[0], degree_star + 1))
        s_mat[:, 0] = 1

        for i in range(1, degree_star + 1):
            s_mat[:, i] = pre_prob_test_temp ** i

        rpr_prob_test[:, seed_num] = s_mat @ poly_star[1]
        t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("The %dth hold-out test has done，%s" % (seed_num, t))
        print("==" * 25, end="\n")

    pd.DataFrame(rpr_prob_test).to_csv("rpr_prob_test.csv")

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--ho_num', metavar='ho_num', type=int, default=None, required=True, help='number of hold out test')
	parser.add_argument('--k_num', metavar='k_num', type=int, default=None, required=True, help='k-fold cross validation')
	parser.add_argument('--bin_num', metavar='bin_num', type=int, default=None, required=True, help='number of bins')
	parser.add_argument('--data_dire', metavar='data_dire', default=None, help='data directory')
	parser.add_argument('--lower_degree', metavar='lower_degree', type=int, default=4, required=True, help='lower degree')
	parser.add_argument('--upper_degree', metavar='upper_degree', type=int, default=15, required=True, help='upper degree')
	parser.add_argument('--lower_lambda', metavar='lower_lambda', type=int, default=-2, required=True, help='lower lambda')
	parser.add_argument('--upper_lambda', metavar='upper_lambda', type=int, default=2, required=True, help='upper_lambda')
	
	args = parser.parse_args()
    
	rpr_cali(args.ho_num, args.k_num, args.bin_num, args.data_dire, args.lower_degree, args.upper_degree, agrs.lower_lambda, args.upper_lambda)
