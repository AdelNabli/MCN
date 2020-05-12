from setuptools import setup, find_packages

setup(
    name="MCN",
    version=0.3,
    packages=find_packages(),
    author="Adel Nabli",
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "matplotlib",
        "torch_geometric",
        "tqdm",
        "tensorboardX",
        #"cplex",
    ],
)
