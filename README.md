# A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve
Python code and data used to reproduce the empirical results of the paper *A neural network approach to the environmental Kuznets curve* by Mikkel Bennedsen, Eric Hillebrand, and Sebastian Jensen, 2023. [doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985).

The code is built with python 3.x (tested with version 3.6.10) and TensorFlow 2.x (tested with version 2.2.0).


## References
* Bennedsen, M., Hillebrand, E., & Jensen, S. (2023). A neural network approach to the environmental Kuznets curve. Energy Economics, 126, Article 106985. [https://doi.org/10.1016/j.eneco.2023.106985](https://doi.org/10.1016/j.eneco.2023.106985).


## Project status
Work in progress.


## Installation
Not yet available through the Python Package Index (PyPi).


## Usage
Save this repository locally (e.g. using Github Desktop), and make sure to have installed all required dependencies. 

The ***Papers*** folder contains the data and source code needed to reproduce all results of the two reference papers.

The ***Functions*** folder contains custom functions used for data preparation and implementation of the neural network-based panel data methodology: 

* The source code for the static neural network model is available in _Static_NN_model.py_.

* The source code for the dynamic neural network model is available in _Dyanmic_NN_model.py_.

 * The source code for the path-dependent neural network model is available in _Path_dependent_NN_model.py_.


## Example
Illustrative example. Function arguments are further explained in Dynamic_NN_model.py and Static_NN_model.py.

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

# NB: x_train, y_train, and pop_train must be aligned to have the same missing values. 
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


## License
Not yet licensed.


## Dependencies
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
