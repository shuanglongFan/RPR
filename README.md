# RPR
Shape restricted polynomial regression

RPR is a flexible method to perform probability calibrion via a polynomial regression. Specifically, this package does the following things:
- peforming cross validation to find optimal degree and lambda using the training data
- fitting polynomial regression with the optimal hyper-parameters using the whole training data
- calibrating the probability on the testing data

Relevent papers were published in:
- [Probability calibration-based prediction of recurrence rate in patients with diffuse large B-cell lymphoma. *BioData* Mining 14, 38(2021).](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00272-9)
- [Applying probability calibration to ensemble methods to predict 2-year mortality in patients with DLBCL. BMC Med Inform Decis Mak 21, 14 (2021)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01354-0)
