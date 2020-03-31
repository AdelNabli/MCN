from setuptools import setup, find_packages

setup(
    name="MCN",
    version=0.1,
    packages=find_packages(where="MCN"),
    package_dir={"": "MCN"},
    author="Adel Nabli",
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "matplotlib",
        "torch_geometric",
        "tqdm",
        "tensorboardX",
    ],
)
