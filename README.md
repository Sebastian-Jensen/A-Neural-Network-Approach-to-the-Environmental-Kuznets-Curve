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


### Example

*x_train* and *y_train* must be specified as dictionaries where the key-value pairs are region names and pandas DataFrames of observations, respectively.

Missing values are accounted for, but *x_train* and *y_train* must be aligned to have the same missing values. 


```python
from Dynamic_NN_model import dynamic_model as Model

print('hello_world')
```

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "pycharm": {
     "name": "#%% Importing libraries\n"
    }
   },
   "outputs": [],
   "source": [
    "import os; os.environ['PYTHONHASHSEED'] = str(0)\n",
    "import sys; sys.path.append('Functions')\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import pandas as pd\n",
    "\n",
    "from fPrepare import fPrepare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "specification = 'dynamic'  # Must be 'static' or 'dynamic'\n",
    "formulation = 'regional'   # Must be 'global', 'regional', or 'national'\n",
    "nodes = (8, 8, 8)          # Must be (x,), (x,y), or (x,y,z)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% Setting choice parameters\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "if specification == 'static':\n",
    "    from Static_NN_model import static_model as Model\n",
    "\n",
    "elif specification == 'dynamic':\n",
    "    from Dynamic_NN_model import dynamic_model as Model\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% Loading model class\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "GDP = pd.read_excel('Data/GDP.xlsx', sheet_name='Python', index_col=0)\n",
    "GDP.sort_index(axis=1, inplace=True)\n",
    "\n",
    "POP = pd.read_excel('Data/Population.xlsx', sheet_name='Python', index_col=0) / 1e6\n",
    "POP.sort_index(axis=1, inplace=True)\n",
    "\n",
    "DEF = pd.read_excel('Data/Deflator.xlsx', sheet_name='Python', index_col=0)\n",
    "DEF.sort_index(axis=1, inplace=True)\n",
    "\n",
    "PPP = pd.read_excel('Data/PPP.xlsx', sheet_name='Python', index_col=0)\n",
    "PPP.sort_index(axis=1, inplace=True)\n",
    "\n",
    "GHG = pd.read_excel('Data/CO2_GCP.xlsx', sheet_name='Python', index_col=0) * 3.664\n",
    "GHG.sort_index(axis=1, inplace=True)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% Loading data\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "gdp_est, ghg_est, pop_est = fPrepare(GDP, POP, DEF, PPP, GHG)\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% Preparing data\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "model = Model(nodes=nodes, x_train=gdp_est, y_train=ghg_est, pop_train=pop_est, formulation=formulation)\n",
    "\n",
    "model.fit(lr=0.001, min_delta=1e-6, patience=100, verbose=False)\n",
    "\n",
    "model.in_sample_predictions()\n",
    "BIC = model.BIC"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

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
