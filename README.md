# RPR
Shape restricted polynomial regression

RPR is a flexible method to perform probability calibrion. Specifically, it does the following things:
- peforming cross validation to find optimal degree and lambda on the training data
- fitting polynomial regression with the optimal hyper-parameters using the whole training data
- calibrating the probability on the testing data
