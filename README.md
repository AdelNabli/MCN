# MCN
Learning to solve the Multilevel Critical Node (MCN) Problem

## Usage
At the moment, it is only possible to train a new neural network on a distribution of instances of the MCN problem.
First, install the package, e.g as following
```
python -m pip install git+https://github.com/AdelNabli/MCN/
```
Then, run the following python script:
```python
# Import the training function
from MCN.MCN_curriculum.train import train_value_net

# Train the neural network
train_value_net(batch_size=20, memory_size=256, lr=1e-3, betas=(0.8,0.9), E=100000,
                target_update=100, h1=128, h2=64, n_heads=3, alpha=0.1, tolerance=0.2,
                n_free_min=4, n_free_max=10, Omega_max=2, Phi_max=2, Lambda_max=2)
```

## Requirements
* [numpy](https://numpy.org/)
* [networkx](https://networkx.github.io/)
* [matplotlib](https://matplotlib.org/)
* [pytorch](https://pytorch.org/)
* [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
* [tensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html)
* [tqdm](https://tqdm.github.io/)
