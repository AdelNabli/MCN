import torch
import pickle
import os
from tqdm import tqdm
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from MCN.utils import generate_random_instance, generate_random_batch_instance, instance_to_torch, InstanceTorch
from MCN.MCN_heur.mcn_heuristic import solve_mcn_heuristic_batch
from MCN.solve_mcn import solve_mcn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCNDataset(Dataset):

    """Create the torch Dataset object"""

    def __init__(self, list_instances):

        self.data = list_instances

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


def collate_fn(list_instances):

    """Given a list of instances, gather them all into a single instance"""

    # Initialize the collated instance
    instances_collated = InstanceTorch(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # Create a batch data object from Pytorch Geometric
    instances_collated.G_torch = Batch.from_data_list(
        [instances.G_torch for instances in list_instances]
    ).to(device)
    # Concatenate all the other parameters
    instances_collated.n_nodes = torch.cat([instances.n_nodes for instances in list_instances])
    instances_collated.Omegas = torch.cat([instances.Omegas for instances in list_instances])
    instances_collated.Phis = torch.cat([instances.Phis for instances in list_instances])
    instances_collated.Lambdas = torch.cat([instances.Lambdas for instances in list_instances])
    instances_collated.Omegas_norm = torch.cat([instances.Omegas_norm for instances in list_instances])
    instances_collated.Phis_norm = torch.cat([instances.Phis_norm for instances in list_instances])
    instances_collated.Lambdas_norm = torch.cat([instances.Lambdas_norm for instances in list_instances])
    instances_collated.J = torch.cat([instances.J for instances in list_instances])
    instances_collated.target = torch.cat([instances.target for instances in list_instances])
    # try if there are 'players' in the instances
    try:
        instances_collated.next_player = torch.cat([instances.next_player for instances in list_instances])
    except:
        instances_collated.next_player = None

    return instances_collated


def load_create_datasets(size_train_data, size_val_data, batch_size, num_workers, n_free_min, n_free_max,
                         d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted, w_max, directed, Budget,
                         list_experts, path_data, solve_exact=False, exact_protection=False, batch_unroll=None):

    """Create or load the training and validation sets.
    Return two dataloaders to access both datasets.
    Dump the datasets in a .gz file in data/train_data and data/val_data"""

    print("\n==========================================================================")
    print("Creating or Loading the Training and Validation sets for Budget = %2d \n" % Budget)

    # Initialize the dataset and number of instances to generate
    data = []
    len_data_train = 0
    total_size = size_train_data + size_val_data
    # If there is a data folder
    if path_data is not None:
        # we check whether there is already a training set
        # corresponding to the budget we want
        path_train_data_budget = os.path.join(path_data, 'train_data', 'data_'+str(Budget)+'.gz')
        # if it's the case, we load it
        if os.path.exists(path_train_data_budget):
            data += pickle.load(open(path_train_data_budget, "rb"))
            len_data_train = len(data)
        # similarly, we check whether there is a validation set available
        path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
        # if it's the case, we load it
        if os.path.exists(path_val_data_budget):
            data += pickle.load(open(path_val_data_budget, "rb"))

    # Update the number of instances that needs to be created
    total_size = total_size - len(data)

    # We create the instances that are currently lacking in the datasets
    # If we need the exact protection, we solve one instance at a time
    if exact_protection:
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
                weighted,
                w_max,
                directed,
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
                exact_protection=exact_protection,
            )
            instance.value = value
            # Transform the instance to a InstanceTorch object
            instance_torch = instance_to_torch(instance)
            # add the instance to the data
            data.append(instance_torch)
    # Else, we can solve batches of instances together
    else:
        # Compute the number of batches necessary to fill the memory
        if batch_unroll is None:
            min_size_instance = n_free_min + Budget
            max_size_instance = n_free_max + Budget
            mean_size_instance = min_size_instance + (max_size_instance - min_size_instance) // 2
            batch_instances = batch_size // mean_size_instance
        else:
            batch_instances = batch_unroll
        n_iterations = total_size // batch_instances + 1 * (total_size % batch_instances > 0)
        for k in tqdm(range(n_iterations)):
            # Sample a batch of random instance
            list_instances = generate_random_batch_instance(
                batch_instances,
                n_free_min,
                n_free_max,
                d_edge_min,
                d_edge_max,
                Omega_max,
                Phi_max,
                Lambda_max,
                Budget,
                weighted,
                w_max,
                directed,
            )
            # Solves the mcn problem for the batch using the heuristic
            values = solve_mcn_heuristic_batch(
                list_experts,
                list_instances,
                Omega_max,
                Phi_max,
                Lambda_max,
            )
            for i in range(batch_instances):
                list_instances[i].value = values[i]
                # Transform the instance to a InstanceTorch object
                instance_torch = instance_to_torch(list_instances[i])
                # add the instance to the data
                data.append(instance_torch)


    # Save the data if there is a change in the dataset
    if len_data_train != size_train_data or total_size > 0:
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
    val_data = MCNDataset(data[size_train_data:size_train_data + size_val_data])
    train_data = MCNDataset(data[:size_train_data])
    train_loader = DataLoader(
        train_data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def generate_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                      weighted, w_max, directed, size_test_set, to_torch=False):

    """Generates a set of random instances that are solved exactly with the MCN_exact algorithm.
    Each budget possible in [1, Omega_max + Phi_max + Lambda_max] is equally represented in
    the test set. The dataset is then dumped in a .gz file inside data\test_data"""

    # Initialize the variables
    Budget_max = Omega_max + Phi_max + Lambda_max
    test_set = []
    if to_torch:
        test_set_torch = []

    print("==========================================================================")
    print("Generates the test set... \n")

    # for all budgets
    for budget in tqdm(range(1, Budget_max + 1)):
        # initialize the budget's instances list
        test_set_budget = []
        if to_torch:
            test_set_budget_torch = []
        for k in range(size_test_set):
            # generate a random instance
            instance_budget_k = generate_random_instance(
                n_free_min,
                n_free_max,
                d_edge_min,
                d_edge_max,
                Omega_max,
                Phi_max,
                Lambda_max,
                weighted=weighted,
                w_max=w_max,
                Budget_target=budget,
                directed=directed,
            )
            G = instance_budget_k.G
            Omega = instance_budget_k.Omega
            Phi = instance_budget_k.Phi
            Lambda = instance_budget_k.Lambda
            J = instance_budget_k.J
            # solve the instance
            value, D, I, P = solve_mcn(G, Omega, Phi, Lambda, J=J, exact=True)
            # save the value, P, D in the Instance object
            instance_budget_k.value = value
            instance_budget_k.D = D
            instance_budget_k.I = I
            instance_budget_k.P = P
            # pushes it to memory
            test_set_budget.append(instance_budget_k)
            # if we want to save the corresponding InstanceTorch
            # to evaluate the training, we stop at Budget_max - 1
            if to_torch and budget < Budget_max:
                instance_budget_k_torch = instance_to_torch(instance_budget_k)
                test_set_budget_torch.append(instance_budget_k_torch)
        test_set.append(test_set_budget)
        if to_torch and budget < Budget_max:
            test_set_torch.append(test_set_budget_torch)

    if not os.path.exists('data'):
        os.mkdir('data')
    folder_name = 'test_data'
    if weighted:
        folder_name += '_w'
    if directed:
        folder_name += '_dir'
    path_test_data = os.path.join('data', folder_name)
    if not os.path.exists(path_test_data):
        os.mkdir(path_test_data)
    # Save the test sets
    file_path = os.path.join(path_test_data, "test_set.gz")
    pickle.dump(test_set, open(file_path, "wb"))
    if to_torch:
        file_path_torch = os.path.join(path_test_data, "test_set_torch.gz")
        pickle.dump(test_set_torch, open(file_path_torch, "wb"))



def load_create_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                         weighted, w_max, directed, size_test_set, path_test_data, batch_size, num_workers):

    """Load or create the test sets and returns a list of Dataloaders to access each test set """

    test_set_generators = []
    if size_test_set > 0 :
        if path_test_data is None:
            generate_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                              weighted, w_max, directed, size_test_set, to_torch=True)
            folder_name = 'test_data'
            if weighted:
                folder_name += '_w'
            if directed:
                folder_name += '_dir'
            path_test_set = os.path.join('data', folder_name, 'test_set_torch.gz')
        else:
            path_test_set = os.path.join(path_test_data, 'test_set_torch.gz')
        # load the test set
        test_set = pickle.load(open(path_test_set, "rb"))
        # create a dataloader object for each dataset in the test set
        for k in range(len(test_set)):
            test_set_k = MCNDataset(test_set[k])
            test_gen_k = DataLoader(
                test_set_k,
                collate_fn=collate_fn,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            test_set_generators.append(test_gen_k)

    return test_set_generators
