# RPR
Shape restricted polynomial regression

RPR is a flexible method to perform probability calibration via a polynomial regression. Specifically, this package does the following things:
- Peforming cross validation to find optimal polynomial degree and regularization constant on the training data
- Fitting polynomial regression with the optimal hyper-parameters using the whole training data
- Calibrating the probability on the testing data and measuring the calibration performance (MSE, ECE, MCE)

Relevent papers were published in:
- [Probability calibration-based prediction of recurrence rate in patients with diffuse large B-cell lymphoma. *BioData Mining* 14, 38(2021).](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00272-9)
- [Applying probability calibration to ensemble methods to predict 2-year mortality in patients with DLBCL. *BMC Med Inform Decis Mak* 21, 14 (2021).](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01354-0)
- [Calibrating Classification Probabilities with Shape-Restricted Polynomial Regression. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41,(8), 1813-1827, 2019.](https://ieeexplore.ieee.org/document/8627976)
