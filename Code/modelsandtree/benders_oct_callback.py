from gurobipy import *
import time
from utilities.utils import Utils


def get_left_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f]
                   for f in master.cat_features if master.data.at[i, f] == 0)

    return lhs


def get_right_exp_integer(master, b, n, i):
    lhs = quicksum(-master.m[i] * master.b[n, f]
                   for f in master.cat_features if master.data.at[i, f] == 1)

    return lhs


def get_target_exp_integer(master, p, beta, n, i):
    label_i = master.data.at[i, master.label]

    if master.mode == "classification":
        lhs = -1 * master.beta[n, label_i]
    elif master.mode == "regression":
        # min (m[i]*p[n] - y[i]*p[n] + beta[n] , m[i]*p[n] + y[i]*p[n] - beta[n])
        if master.m[i] * p[n] - label_i * p[n] + beta[n, 1] < master.m[i] * p[n] + label_i * p[n] - beta[n, 1]:
            lhs = -1 * (master.m[i] * master.p[n] -
                        label_i * master.p[n] + master.beta[n, 1])
        else:
            lhs = -1 * (master.m[i] * master.p[n] +
                        label_i * master.p[n] - master.beta[n, 1])

    return lhs


def get_cut_integer(master, b, p, beta, left, right, target, i):
    lhs = LinExpr(0) + master.g[i]
    for n in left:
        tmp_lhs = get_left_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in right:
        tmp_lhs = get_right_exp_integer(master, b, n, i)
        lhs = lhs + tmp_lhs

    for n in target:
        tmp_lhs = get_target_exp_integer(master, p, beta, n, i)
        lhs = lhs + tmp_lhs

    return lhs


def subproblem(master, b, p, beta, i):
    label_i = master.data.at[i, master.label]
    current = 1
    right = []
    left = []
    target = []
    subproblem_value = 0

    while True:
        pruned, branching, selected_feature, terminal, current_value = Utils(
        ).get_node_status(master, b, beta, p, current)
        if terminal:
            target.append(current)
            if current in master.tree.Nodes:
                left.append(current)
                right.append(current)
            if master.mode == "regression":
                subproblem_value = master.m[i] - abs(current_value - label_i)
            elif master.mode == "classification" and beta[current, label_i] > 0.5:
                subproblem_value = 1
            break
        elif branching:
            if master.data.at[i, selected_feature] == 1:  # going right on the branch
                left.append(current)
                target.append(current)
                current = master.tree.get_right_children(current)
            else:  # going left on the branch
                right.append(current)
                target.append(current)
                current = master.tree.get_left_children(current)

    return subproblem_value, left, right, target


##########################################################
# Defining the callback function
###########################################################
def mycallback(model, where):
    '''
    This function is called by gurobi at every node through the branch-&-bound tree while we solve the model.
    Using the argument "where" we can see where the callback has been called. We are specifically interested at nodes
    where we get an integer solution for the master problem.
    When we get an integer solution for b and p, for every datapoint we solve the subproblem which is a minimum cut and
    check if g[i] <= value of subproblem[i]. If this is violated we add the corresponding benders constraint as lazy
    constraint to the master problem and proceed. Whenever we have no violated constraint! It means that we have found
    the optimal solution.
    :param model: the gurobi model we are solving.
    :param where: the node where the callback function is called from
    :return:
    '''
    data_train = model._master.data
    mode = model._master.mode

    local_eps = 0.0001
    if where == GRB.Callback.MIPSOL:
        func_start_time = time.time()
        model._callback_counter_integer += 1
        # we need the value of b,w and g
        g = model.cbGetSolution(model._vars_g)
        b = model.cbGetSolution(model._vars_b)
        p = model.cbGetSolution(model._vars_p)
        beta = model.cbGetSolution(model._vars_beta)

        added_cut = 0
        # We only want indices that g_i is one!
        for i in data_train.index:
            if mode == "classification":
                g_threshold = 0.5
            elif mode == "regression":
                g_threshold = 0
            if g[i] > g_threshold:
                subproblem_value, left, right, target = subproblem(
                    model._master, b, p, beta, i)
                if mode == "classification" and subproblem_value == 0:
                    added_cut = 1
                    lhs = get_cut_integer(
                        model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)
                elif mode == "regression" and ((subproblem_value + local_eps) < g[i]):
                    added_cut = 1
                    lhs = get_cut_integer(
                        model._master, b, p, beta, left, right, target, i)
                    model.cbLazy(lhs <= 0)

        func_end_time = time.time()
        func_time = func_end_time - func_start_time
        # print(model._callback_counter)
        model._total_callback_time_integer += func_time
        if added_cut == 1:
            model._callback_counter_integer_success += 1
            model._total_callback_time_integer_success += func_time
