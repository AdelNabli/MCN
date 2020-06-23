# MCN
Learning to solve the Multilevel Critical Node (MCN) Problem

## Setup
Install the package, e.g as following
```
python -m pip install git+https://github.com/AdelNabli/MCN/
```

## Requirements
* [numpy](https://numpy.org/)
* [networkx](https://networkx.github.io/)
* [matplotlib](https://matplotlib.org/) (only used for plotting the graphs)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) (only used to display tables of results)
* [pytorch](https://pytorch.org/)
* [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (To use our pre-trained models, a version v1.4.x is necessary as changes in the source code of some *Graph Neural Networks* happened in v1.5.0)
* [tensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html)
* [tqdm](https://tqdm.github.io/)
* [cplex](https://www.ibm.com/analytics/cplex-optimizer) (only necessary if the exact algorithm is used)

## Usage
There are 3 main tasks supported:
* Train a neural network to produce a pool of 'expert nets' in order to solve the MCN problem on a distribution of instances
* Solve an instance of the MCN problem, either using an exact method or heuristically with the trained experts
* Evaluate the performances of the trained experts on a test set of exactly solved MCN instances

An example of how to perform each of these tasks is given in the [Notebook](https://github.com/AdelNabli/MCN/blob/master/Usages.ipynb)

## Acknowledgement
The MCN problem was introduced in **A. Baggio, M. Carvalho, A. Lodi, A. Tramontani**, ["Multilevel Approaches for the Critical Node Problem"]( http://cerc-datascience.polymtl.ca/wp-content/uploads/2017/11/Technical-Report_DS4DM-2017-012.pdf), 2018. The exact method is a simple implementation of the algorithm described in this paper. Our implementation is based on the original code found in the following Github repository: [mxmmargarida/Critical-Node-Problem](https://github.com/mxmmargarida/Critical-Node-Problem)
