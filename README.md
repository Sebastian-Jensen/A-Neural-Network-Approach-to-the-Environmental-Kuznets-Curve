# A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve
Python code and data used to reproduce the empirical results of the paper *A neural network approach to the environmental Kuznets curve* by Mikkel Bennedsen, Eric Hillebrand, and Sebastian Jensen, 2023. [doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985). We propose a framework that contains two distinct model specifications:

* The ***static model*** contains country and time fixed effects in addition to a neural network regression component.

* The ***dynamic model*** uses a time variable as an additional input into the neural network component in place of time fixed effects.


## Usage
Save this repository locally (e.g. using Github Desktop) and ensure all required dependencies have been installed. 

The folder ***Empirical Analysis*** contains scripts for model estimation and reproduction of empirical results (figures and tables).

The folder ***Functions*** contains functions for data preparation and implementation of the neural network-based panel data methodology. 

The folder ***Data*** contains the data used in the paper.

The folders ***Benchmark models***, ***BIC***, ***Model Parameters***, ***Model parameters sd***, and ***Squared residuals*** contains the parameter estimates used in the paper and are overwritten by the scripts in the folder ***Empirical Analysis***.


## Example
This example demonstrates how to use the _dynamic model_. The _static model_ is used similarly.

### Importing model
```python
from Dynamic_NN_model import dynamic_model as Model
#from Static_NN_model import static_model as Model 
```

### Setting choice parameters
```python
formulation = 'regional'   # Must be 'global', 'regional', or 'national'
nodes = (8, 8, 8)          # Must be (x,), (x,y), or (x,y,z)
```

### Estimating model
```python
model = Model(nodes=nodes, x_train=x_train, y_train=y_train, pop_train=pop_train, formulation=formulation)
model.fit(lr=0.001, min_delta=1e-6, patience=100, verbose=False)

# x_train:    dict of dataframes of input data (time periods x countries) with a key for each region.
# y_train:    dict of dataframes of target data (time periods x countries) with a key for each region.
# pop_train:  dict of dataframes of population data (time periods x countries) with a key for each region.

# NB: x_train, y_train, and pop_train must be aligned to have the same missing values. pop_train is used for transforming the data (log per capita transformation). 
```

### Predictions
```python
y_pred = model.predict(x_test, region, time_test)

# x_test:     (-1,1) array of input data.
# region:     str identifying the region to be used for making predictions.
# time_test:  (-1,1) array of input data.
```

### Visualizing output from dynamic model
<p float="left">
  <img src="/Figures examples/f_OECD_with_ben.png" width="19%" />
  <img src="/Figures examples/f_REF_with_ben.png" width="19%" />
  <img src="/Figures examples/f_Asia_with_ben.png" width="19%" />
  <img src="/Figures examples/f_MAF_with_ben.png" width="19%" />
  <img src="/Figures examples/f_LAM_with_ben.png" width="19%" />
</p>


## References
* Bennedsen, M., Hillebrand, E., & Jensen, S. (2023). A neural network approach to the environmental Kuznets curve. Energy Economics, 126, Article 106985. [https://doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985).


## Dependencies
python 3.x (tested with version 3.6.10) 

TensorFlow 2.x (tested with version 2.2.0).

numpy (tested with version 1.19.4)

scipy (tested with version 1.4.1)

pandas (tested with version 1.1.1)

tensorflow (tested with version 2.2.0)

keras (tested with version 2.4.3)

matplotlib (tested with version 3.1.1)

seaborn (tested with version 0.9.0)


## Contact
Sebastian Jensen

Mail: smjensen@econ.au.dk

Website: https://sites.google.com/view/sebastianjensen

GitHub profile: https://github.com/Sebastian-Jensen

Project link: https://github.com/Sebastian-Jensen/A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve


## Co-authors
[Eric Hillebrand](https://sites.google.com/site/erichillebrand)

[Mikkel Bennedsen](https://sites.google.com/site/mbennedsen/home)
