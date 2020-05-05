import cplex


def solve_defender(I, G, Lambda):

    r"""Solve the Defender problem under attack I with budget Lambda.

    Parameters:
    ----------
    I: list of ints,
       contains the ids of the attacked nodes
    G: networkx graph
    Lambda: int,
            protection budget

    Returns:
    -------
    P: list (size=Lambda),
       contains the ids of the nodes to protect
    S: list,
       contains the saved nodes
    value: int,
           number of saved nodes with the attack I"""

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
    Defender = cplex.Cplex()
    ## set the output stream
    Defender.set_log_stream(None)
    Defender.set_error_stream(None)
    Defender.set_warning_stream(None)
    Defender.set_results_stream(None)
    ## set the number of threads to 1
    Defender.parameters.threads.set(1)
    ## set the objective direction to "maximize"
    Defender.objective.set_sense(Defender.objective.sense.maximize)

    # Set the variables

    ## alpha_v: There are |V| continuous variables
    for v in V:
        Defender.variables.add(
            obj=[w[v]], types=[Defender.variables.type.continuous], names=["alpha_%d" % v]
        )
    ## x_v: There are |V| binary variables
    for v in V:
        Defender.variables.add(
            obj=[0], types=[Defender.variables.type.binary], names=["x_%d" % v]
        )

    # Set the constraints

    ## sum_v x_v < Lambda
    Defender.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["x_%d" % v for v in V], val=[1] * len(V))],
        senses=["L"],
        rhs=[Lambda],
    )
    ## alpha_v < 1-y_v for all v
    for v in V:
        Defender.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["alpha_%d" % v], val=[1.0])],
            senses=["L"],
            rhs=[0.0 if v in I else 1.0],
        )
    ## alpha_v < alpha_u + x_v for all (u,v) in E
    for (u, v) in E:
        Defender.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=["alpha_%d" % v, "x_%d" % v, "alpha_%d" % u],
                    val=[1.0, -1.0, -1.0],
                )
            ],
            senses=["L"],
            rhs=[0.0],
        )

    # Solve the problem

    Defender.solve()
    solution = Defender.solution
    ## Get the number of saved nodes
    value = int(round(solution.get_objective_value()))
    ## Get the nodes to defend
    P = [v for v in V if solution.get_values("x_%d" % v) > 0.9]
    ## Get the saved nodes
    S = [v for v in V if solution.get_values("alpha_%d" % v) > 0.9]

    return (value, S, P)
