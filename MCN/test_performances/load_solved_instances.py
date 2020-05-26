import os
import numpy as np
import networkx as nx
import re
from MCN.utils import Instance
from MCN.solve_mcn import solve_mcn
from MCN.test_performances.optimality_gap import opt_gap
from tqdm import tqdm


def recover_instances(folder_name):

    instances = dict()
    instances[20] = []
    instances[40] = []
    instances[60] = []
    instances[80] = []
    instances[100] = []

    # gather the list of directory
    list_dir = os.listdir(folder_name)
    for dir in list_dir:
        numbers_in_dir = re.findall('[0-9]{1,3}', dir)
        if 'tree' not in dir and len(numbers_in_dir) == 5:
            # the nb of nodes is the 2nd number in the name of the dir
            n_nodes = int(numbers_in_dir[1])
            Omega = int(numbers_in_dir[2])
            Phi = int(numbers_in_dir[3])
            Lambda = int(numbers_in_dir[4])
            path_instances = os.path.join(folder_name, dir)
            list_files = os.listdir(path_instances)
            for path_file in list_files:
                path = os.path.join(path_instances, path_file)
                f = open(path, "r")
                failed = False
                for line in f:
                    if 'V =' in line:
                        V = np.array(re.findall('[0-9]{1,2}', line)).astype('int')
                    elif 'A =' in line:
                        A = []
                        list_couples = re.findall('([0-9]{1,2}, [0-9]{1,2})',line)
                        for couple in list_couples:
                            list_uv = re.findall('[0-9]{1,2}', couple)
                            A.append((int(list_uv[0]), int(list_uv[1])))
                    elif '#opt' in line and '#optDA' not in line:
                        value = np.array(re.findall('[0-9]{1,2}', line)).astype('int')[0]
                    elif '#fail' in line and 'yes' in line:
                        failed = True
                if not failed:
                    G = nx.DiGraph()
                    G.add_nodes_from(V)
                    G.add_edges_from(A)
                    instance = Instance(G, Omega, Phi, Lambda, [], value)
                    instances[n_nodes].append(instance)

    return instances


def compute_opt_gap(dict_instances, Omega_max, Phi_max, Lambda_max, list_experts):

    val_exact = dict()
    val_exact[20] = []
    val_exact[40] = []
    val_exact[60] = []
    val_exact[80] = []
    val_exact[100] = []

    val_heur = dict()
    val_heur[20] = []
    val_heur[40] = []
    val_heur[60] = []
    val_heur[80] = []
    val_heur[100] = []

    opt_gaps = []

    for n_nodes in dict_instances.keys():

        instances_n = dict_instances[n_nodes]
        for instance in tqdm(instances_n):
            value_heuristic, D_heur, I_heur, P_heur = solve_mcn(instance.G, instance.Omega, instance.Phi,
                                                                instance.Lambda,
                                                                J=instance.J, Omega_max=Omega_max, Phi_max=Phi_max,
                                                                Lambda_max=Lambda_max, exact=False,
                                                                list_experts=list_experts)
            val_exact[n_nodes].append(instance.value)
            val_heur[n_nodes].append(value_heuristic)

        opt_gap_n = opt_gap(val_exact[n_nodes], val_heur[n_nodes])
        opt_gaps.append(opt_gap_n)
        print(opt_gap_n)

    return opt_gaps, val_exact, val_heur