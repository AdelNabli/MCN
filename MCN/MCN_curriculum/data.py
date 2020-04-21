import torch
import pickle
import os
from tqdm import tqdm
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from MCN.utils import generate_random_instance, instance_to_torch, InstanceTorch
from MCN.solve_mcn import solve_mcn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCNDataset(Dataset):

    def __init__(self, list_instances):

        self.data = list_instances

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


def collate_fn(list_instances):

    # Initialize the collated instance
    instances_collated = InstanceTorch(0, 0, 0, 0, 0, 0, 0, 0, 0)
    # Create a batch data object from Pytorch Geometric
    instances_collated.G_torch = Batch.from_data_list(
        [instances.G_torch for instances in list_instances]
    ).to(device)
    # Concatenate all the other parameters
    instances_collated.Omegas = torch.cat([instances.Omegas for instances in list_instances])
    instances_collated.Phis = torch.cat([instances.Phis for instances in list_instances])
    instances_collated.Lambdas = torch.cat([instances.Lambdas for instances in list_instances])
    instances_collated.J = torch.cat([instances.J for instances in list_instances])
    instances_collated.saved_nodes = torch.cat([instances.saved_nodes for instances in list_instances])
    instances_collated.infected_nodes = torch.cat([instances.infected_nodes for instances in list_instances])
    instances_collated.size_connected = torch.cat([instances.size_connected for instances in list_instances])
    instances_collated.target = torch.cat([instances.target for instances in list_instances])

    return instances_collated


def load_create_datasets(size_train_data, size_val_data, batch_size, num_workers, n_free_min, n_free_max,
                         d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, Budget,
                         list_experts, path_data, solve_exact=False):

    print("\n==========================================================================")
    print("Creating or Loading the Training and Validation sets for Budget = %2d \n" % Budget)

    # Initialize the dataset and number of instances to generate
    data = []
    total_size = size_train_data + size_val_data
    # If there is a data folder
    if path_data is not None:
        # we check whether there is already a training set
        # corresponding to the budget we want
        path_train_data_budget = os.path.join(path_data, 'train_data', 'data_'+str(Budget)+'.gz')
        # if it's the case, we load it
        if os.path.exists(path_train_data_budget):
            data += pickle.load(open(path_train_data_budget, "rb"))
        # similarly, we check whether there is a validation set available
        path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
        # if it's the case, we load it
        if os.path.exists(path_val_data_budget):
            data += pickle.load(open(path_val_data_budget, "rb"))

    # Update the number of instances that needs to be created
    total_size = total_size - len(data)

    # We create the instances that are currently lacking in the datasets
    for k in tqdm(range(total_size)):
        # Sample a random instance
        instance = generate_random_instance(
            n_free_min,
            n_free_max,
            d_edge_min,
            d_edge_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            Budget,
        )
        # Solves the mcn problem
        value, _, _, _ = solve_mcn(
            instance.G,
            instance.Omega,
            instance.Phi,
            instance.Lambda,
            J=instance.J,
            Omega_max=Omega_max,
            Phi_max=Phi_max,
            Lambda_max=Lambda_max,
            exact=solve_exact,
            list_experts=list_experts,
        )
        instance.value = value
        # Transform the instance to a InstanceTorch object
        instance_torch = instance_to_torch(instance)
        # add the instance to the data
        data.append(instance_torch)

    # Save the data
    if path_data is None:
        path_data = 'data'
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path_train = os.path.join(path_data, 'train_data')
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    path_val = os.path.join(path_data, 'val_data')
    if not os.path.exists(path_val):
        os.mkdir(path_val)
    path_train_data_budget = os.path.join(path_train, 'data_' + str(Budget) + '.gz')
    path_val_data_budget = os.path.join(path_val, 'data_' + str(Budget) + '.gz')
    pickle.dump(data[:size_train_data], open(path_train_data_budget, "wb"))
    pickle.dump(data[size_train_data:], open(path_val_data_budget, "wb"))
    print("\nSaved datasets in " + path_data, '\n')

    # Create the datasets used during training and validation
    val_data = collate_fn(data[size_train_data:size_train_data + size_val_data])
    train_data = MCNDataset(data[:size_train_data])
    train_loader = DataLoader(
        train_data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_data
