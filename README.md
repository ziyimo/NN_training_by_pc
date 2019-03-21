## Training feedforward neural net in a predictive coding framework

#### Feb-Mar 2019

Model from [Whittington and Bogacz (2017)](https://doi.org/10.1162/NECO_a_00949); part of the code adapted from [ModelDB Accession: 218084](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=218084#tabs-1).

__Usage__

```bash
$ ./run_MATLAB.sh param_file_path NN_arch_path data_path
```

__param_file__

Contains hyperparameters of the model, tab-delimited with a header. See example below:
```
act_type 	l_rate 	beta1 	numint_its 	epochs 	d_rate 	int_step 	lr_decay 	buf_win 	ADAPTIT_ri 	ADAPTIT_rs 	ADAPTIT_w
tanh    	0.9    	0.75   	100        	2500   	0      	0.1      	1        	10      	4          	2          	1000
```

* `act_type`: Type of activation function, can be `lin`, `reclin`, `tanh` or `sig`.
* `l_rate`: initial learning rate
* `beta1`: parameter for momentum
* `numint_its`: initial # of iterations for numerical integration of the dynamical system
* `epochs`: # of training epochs
* `d_rate`: rate of weight decay (regularization)
* `int_step`: initial step of numerical integration of the dynamical system
* `lr_decay`: rate of decay for learning rate (learning rate is reduced when cost increases after an epoch)
* `buf_win`: buffer window for learning rate and numerical integration step/iteration update schedule
* `ADAPTIT_ri`: growth rate for `numint_its` (i.e. `numint_its = numint_its*ADAPTIT_ri;`)
* `ADAPTIT_rs`: decay rate for `int_step` (i.e. `int_step = int_step/ADAPTIT_rs;`)
* `ADAPTIT_w`: window for mandatory update of integration step/iteration (usually only updated when cost increases after an epoch)

__NN_arch__

Contains the architecture of the feedforward network, space delimited. The first entry is the size of input layer (i.e. dimensionality of input data) and the last __two__ entries are the size of the output layer (the output layer has to be repeated twice due to a technicality). See example below:

```
400 30 10 10
```

__data__

`.mat` format file with training data `X` (# of samples $\times$ dimension of data) and label `y` (a column vector).

__Output__

* `[time_stamp]_best.mat`: Saves the best model (in terms of training accuracy) so far, continuously updated during training.
* `[time_stamp].mat`: Saves the model learnt at the end of all training iterations, as well as cost history to plot training curve.
* `[time_stamp].txt`: Prints out the model parameters at the beginning and prints the cost and accuracy of the model after each trainin epoch.