import cplex


def solve_mip(Q, G, Lambda, Omega):

    r"""Find the best vaccination strategy for the subset of attacks Q and with budgets Lambda, Omega.

    Parameters:
    ----------
    Q: list of lists (size=|Q| x Phi),
       list of the attacks considered.
       each attack is a list of nodes.
    G: networkx graph,
    Lambda: int,
            protection budget
    Omega: int,
           vaccination budget

    Returns:
    -------
    D: list (size=Omega),
       nodes to vaccinate
    best: int,
          number of saved nodes with the vaccination D"""

    # Gather the list of nodes and edges of the graph
    V = list(G.nodes())
    E = list(G.edges())
    # Create a dict of weights
    w = dict()
    for v in V:
        # if the graph is weighted, gather the weights
        if 'weight' in G.nodes[v].keys():
            w[v] = float(G.nodes[v]['weight'])
        # else, all weights are 1
        else:
            w[v] = 1.0

    # Initialize the optimization problem

    ## create the cplex model
    onelvlMIP = cplex.Cplex()
    ## set the output stream
    onelvlMIP.set_log_stream(None)
    onelvlMIP.set_error_stream(None)
    onelvlMIP.set_warning_stream(None)
    onelvlMIP.set_results_stream(None)
    ## set the number of threads to 1
    onelvlMIP.parameters.threads.set(1)
    ## set the objective direction to "maximize"
    onelvlMIP.objective.set_sense(onelvlMIP.objective.sense.maximize)

    # Set the variables

    ## Delta: is continuous and its objective is 1 as we want to maximize it
    onelvlMIP.variables.add(
        obj=[1], types=[onelvlMIP.variables.type.continuous], names=["Delta"]
    )
    ## z_v: There are |V| binary variable z_v, they do not appear in the objective
    ##      so their objective is 0
    for v in V:
        onelvlMIP.variables.add(
            obj=[0], types=[onelvlMIP.variables.type.binary], names=["z_%d" % v]
        )
    ## alpha_v(y): There are |V|*|Q| continuous variables, they do not appear
    ##             in the objective so their objective is 0
    for v in V:
        for id_y in range(len(Q)):
            onelvlMIP.variables.add(
                obj=[0],
                types=[onelvlMIP.variables.type.continuous],
                names=["alpha_%d_%d" % (v, id_y)],
            )
    ## x_v(y): There are |V|*|Q| binary variables, they do not appear
    ##         in the objective so their objective is 0
    for v in V:
        for id_y in range(len(Q)):
            onelvlMIP.variables.add(
                obj=[0],
                types=[onelvlMIP.variables.type.binary],
                names=["x_%d_%d" % (v, id_y)],
            )

    # Set the constraints

    ## Delta < sum_v alpha_v(y) for all y in Q
    for id_y in range(len(Q)):
        onelvlMIP.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=["Delta"] + ["alpha_%d_%d" % (v, id_y) for v in V],
                    val=[1.0] + [-w[v] for v in V],
                )
            ],
            senses=["L"],
            rhs=[0.0],
        )
    ## sum_v z_v < Omega
    onelvlMIP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["z_%d" % v for v in V], val=[1] * len(V))],
        senses=["L"],
        rhs=[Omega],
    )
    ## sum_v x_v(y) < Lambda for all y in Q
    for id_y in range(len(Q)):
        onelvlMIP.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=["x_%d_%d" % (v, id_y) for v in V], val=[1.0] * len(V)
                )
            ],
            senses=["L"],
            rhs=[Lambda],
        )
    ## alpha_v(y) < 1 + z_v - y_v for all v in V and y in Q
    for v in V:
        for id_y in range(len(Q)):
            onelvlMIP.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=["alpha_%d_%d" % (v, id_y), "z_%d" % v], val=[1.0, -1.0]
                    )
                ],
                senses=["L"],
                rhs=[0.0 if v in Q[id_y] else 1.0],
            )
    ## alpha_v(y) < alpha_u(y) + x_v(y) + z_v for all (u,v) in E and y in Q
    for (u, v) in E:
        for id_y in range(len(Q)):
            onelvlMIP.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=[
                            "alpha_%d_%d" % (v, id_y),
                            "x_%d_%d" % (v, id_y),
                            "alpha_%d_%d" % (u, id_y),
                            "z_%d" % v,
                        ],
                        val=[1.0, -1.0, -1.0, -1.0],
                    )
                ],
                senses=["L"],
                rhs=[0.0],
            )
    ## 0 < alpha_v(y) < 1 for all v for all y
    for v in V:
        for id_y in range(len(Q)):
            onelvlMIP.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["alpha_%d_%d" % (v, id_y)], val=[1.0])],
                senses=["L"],
                rhs=[1.0],
            )
            onelvlMIP.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["alpha_%d_%d" % (v, id_y)], val=[1.0])],
                senses=["G"],
                rhs=[0.0],
            )

    # Solve the problem

    onelvlMIP.solve()
    solution = onelvlMIP.solution
    ## Get the number of saved nodes
    best = int(round(solution.get_objective_value()))
    ## Get the nodes to vaccinate
    D = [v for v in V if solution.get_values("z_%d" % v) > 0.9]

    return (best, D)
