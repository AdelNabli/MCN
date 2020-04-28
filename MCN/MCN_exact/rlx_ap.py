import cplex


def solve_rlxAP(S, V, E, Lambda, Phi, J=[]):

    r"""Solve the relaxed Attack-Protect problem with budgets Lambda, Phi.

    Parameters:
    ----------
    S: list of lists,
       S[t] being the list of saved nodes at iteration t
    V: list of ints,
       list of the vertices of the graph
    E: list of tuples of ints,
       list of the edges of the graph
       if (v,u) \in E, then (u,v) must be too
    Lambda: int,
            protection budget
    Phi: int,
         attack budget
    J: list of ints,
       list of the vertices already attacked

    Returns:
    -------
    I: list of ints (size=Phi),
       ids of the nodes to attack
    value: int,
           number of saved nodes with the attack I"""

    # Initialize the optimization problem

    ## create the cplex model
    rlxAP = cplex.Cplex()
    ## set the output stream
    rlxAP.set_log_stream(None)
    rlxAP.set_error_stream(None)
    rlxAP.set_warning_stream(None)
    rlxAP.set_results_stream(None)
    ## set the number of threads to 1
    rlxAP.parameters.threads.set(1)
    ## set the objective direction to "minimize"
    rlxAP.objective.set_sense(rlxAP.objective.sense.minimize)

    # Set the variables

    ## p: is continuous and its objective is Lambda
    rlxAP.variables.add(obj = [Lambda],
                        types = [rlxAP.variables.type.continuous],
                        names = ["p"])
    ## y_v: There are |V| binary variables, with objective 0
    for v in V:
        rlxAP.variables.add(obj = [0],
                            types = [rlxAP.variables.type.binary],
                            names = ["y_%d"%v])
    ## h_v: There are |V| continuous variables, with objective 0
    for v in V:
        rlxAP.variables.add(obj = [0],
                            types = [rlxAP.variables.type.continuous],
                            names = ["h_%d"%v])
    ## gamma_v: There are |V| continuous variables, with objective 1
    for v in V:
        rlxAP.variables.add(obj = [1],
                            types = [rlxAP.variables.type.continuous],
                            names = ["gamma_%d"%v])
    ## q(u,v): There are |E| continuous variables, with objective 0
    for (u,v) in E:
        rlxAP.variables.add(obj = [0],
                            types = [rlxAP.variables.type.continuous],
                            names = ["q_%d_%d"%(u,v)])

    # Set the constraints

    ## sum_v y_v < Phi
    rlxAP.linear_constraints.add(
        lin_expr = [cplex.SparsePair(ind = ["y_%d"%v for v in V],
                                     val = [1]*len(V))],
        senses = ["L"],
        rhs = [Phi])
    ## y_v < 1-z_v for all v
    for v in V:
        rlxAP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["y_%d" % v], val=[1.0])],
            senses=["L"],
            rhs=[0.0 if v in J else 1.0],
        )
    ## h_v + sum_(u,v) q_(u,v) - q_(v,u) > 1 for all v in V
    for v in V:
        edges_in = ["q_%d_%d"%(u,v) for (u,v1) in E if v1==v]
        edges_out = ["q_%d_%d"%(v,u) for (v1,u) in E if v1==v]
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(
                ind = ["h_%d"%v] + edges_in + edges_out,
                val = [1.0] + [1.0]*len(edges_in) + [-1.0]*len(edges_out))],
            senses = ["G"],
            rhs = [1.0])
    ## p - sum_(u,v) q_(u,v) > 0 for all v in V
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(
                ind = ["p"] + edges_in,
                val = [1.0] + [-1.0]*len(edges_in))],
            senses = ["G"],
            rhs = [0.0])
    ## gamma_v + |V|*y_v - h_v > -|V|*z_v for all v in V
    for v in V:
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(
                ind = ["gamma_%d"%v, "y_%d"%v, "h_%d"%v],
                val = [1.0, len(V), -1.0])],
            senses = ["G"],
            rhs = [-len(V) if v in J else 0.0])
    ## p > 0
    rlxAP.linear_constraints.add(
        lin_expr = [cplex.SparsePair(ind = ["p"], val = [1.0])],
        senses = ["G"],
        rhs = [0.0])
    ## h_v > 0 for all v in V
    for v in V:
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = ["h_%d"%v], val = [1.0])],
            senses = ["G"],
            rhs = [0.0])
    ## gamma_v > 0 for all v in V
    for v in V:
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = ["gamma_%d"%v], val = [1.0])],
            senses = ["G"],
            rhs = [0.0])
    ## q_(u,v) > 0 for all (u,v) in E
    for (u,v) in E:
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = ["q_%d_%d"%(u,v)], val = [1.0])],
            senses = ["G"],
            rhs = [0.0])
    ## sum_{v in S_t} y_v > 1  for all t
    for t in range(len(S)):
        rlxAP.linear_constraints.add(
            lin_expr = [cplex.SparsePair(
                ind = ["y_%d"%v for v in S[t]],
                val = [1]*len(S[t]))],
            senses = ["G"],
            rhs = [1])

    # Solve the problem

    rlxAP.solve()
    solution = rlxAP.solution
    ## Get the number of saved nodes
    value = int(round(solution.get_objective_value()))
    ## Get the nodes to attack
    I = [v for v in V if solution.get_values("y_%d"%v) > 0.9]

    return(value, I)
