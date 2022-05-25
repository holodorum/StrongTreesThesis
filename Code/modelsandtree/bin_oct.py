'''
This module formulate the OCT problem in gurobipy.
'''
from collections import namedtuple
import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
from scipy import stats


class BinOCT:
    def __init__(self, data, label, tree, _lambda, time_limit, mip_gap, mode):
        '''

        :param data: The training data
        :param label: Name of the column representing the class label
        :param tree: Tree object
        :param _lambda: The regularization parameter in the objective
        :param time_limit: The given time limit for solving the MIP
        :param mode: Regression vs Classification
        '''
        self.mode = mode

        self.data = data
        self.datapoints = data.index
        self.label = label

        if self.mode == "classification":
            self.labels = data[label].unique()
        elif self.mode == "regression":
            self.labels = [1]
        '''
        cat_features is the set of all categorical features.
        reg_features is the set of all features used for the linear regression prediction model in the leaves.
        '''
        self.cat_features = self.data.columns[self.data.columns != self.label]
        # self.reg_features = None
        # self.num_of_reg_features = 1

        self.tree = tree
        self._lambda = _lambda

        # Decision Variables
        self.b = 0
        self.p = 0
        self.beta = 0
        self.zeta = 0
        self.z = 0

        # parameters
        self.m = {}
        for i in self.datapoints:
            self.m[i] = 1

        if self.mode == "regression":
            for i in self.datapoints:
                y_i = self.data.at[i, self.label]
                self.m[i] = max(y_i, 1 - y_i)

        # thresholds
        self.thresholds = self._getThresholds(
            self.data.loc[:, self.data.columns != self.label], self.data.loc[:, self.label])
        self.thresholds = pd.Series(self.thresholds, index=self.cat_features)
        self.bin_num = int(
            np.ceil(np.log2(max([len(threshold) for threshold in self.thresholds]))))

        # Gurobi model
        self.model = Model('binOCT')
        # # The cuts we add in the callback function would be treated as lazy constraints
        # self.model.params.LazyConstraints = 1
        '''
        To compare all approaches in a fair setting we limit the solver to use only one thread to merely evaluate
        the strength of the formulation.
        '''
        self.model.params.Threads = 1
        self.model.params.TimeLimit = time_limit
        self.model.params.MIPGAP = mip_gap

        '''
        The following variables are used for the Benders problem to keep track of the times we call the callback.

        - counter_integer tracks number of times we call the callback from an integer node in the branch-&-bound tree
            - time_integer tracks the associated time spent in the callback for these calls
        - counter_general tracks number of times we call the callback from a non-integer node in the branch-&-bound tree
            - time_general tracks the associated time spent in the callback for these calls

        the ones ending with success are related to success calls. By success we mean ending up adding a lazy constraint
        to the model


        '''
        self.model._total_callback_time_integer = 0
        self.model._total_callback_time_integer_success = 0

        self.model._total_callback_time_general = 0
        self.model._total_callback_time_general_success = 0

        self.model._callback_counter_integer = 0
        self.model._callback_counter_integer_success = 0

        self.model._callback_counter_general = 0
        self.model._callback_counter_general_success = 0

    ###########################################################
    # Create the MIP formulation
    ###########################################################

    def create_primal_problem(self):
        '''
        This function create and return a gurobi model formulating the binOCT problem
        :return:  gurobi model object with the OCT formulation
        '''
        """
        build MIP formulation for Optimal Decision Tree
        """

        # model sense
        self.modelSense = GRB.MINIMIZE

        # variables
        self.e = self.model.addVars(
            self.tree.Leaves, self.labels, vtype=GRB.CONTINUOUS, name='e')  # leaf node misclassified
        self.f = self.model.addVars(
            self.tree.Nodes, self.cat_features, vtype=GRB.BINARY, name='f')  # splitting feature
        self.l = self.model.addVars(
            self.datapoints, self.tree.Leaves, vtype=GRB.CONTINUOUS, name='z')  # leaf node assignment
        self.p = self.model.addVars(
            self.tree.Leaves, self.labels, vtype=GRB.BINARY, name='p')  # node prediction
        self.q = self.model.addVars(
            self.tree.Nodes, self.bin_num, vtype=GRB.BINARY, name='q')  # threshold selection

        # objective function
        self.model.setObjective(self.e.sum())

        # constraints
        self.model.addConstrs(self.f.sum(t, '*') == 1 for t in self.tree.Nodes)
        self.model.addConstrs(self.l.sum(r, '*') == 1 for r in self.datapoints)
        self.model.addConstrs(self.p.sum(t, '*') ==
                              1 for t in self.tree.Leaves)

        for j in self.cat_features:
            for b in self._getBins(0, len(self.thresholds[j])-1):
                # left
                lb, ub = self.thresholds[j][b.ur[0]
                                            ], self.thresholds[j][b.ur[1]]
                M = np.sum(np.logical_and(
                    self.data.loc[:, j] >= lb, self.data.loc[:, j] <= ub))
                for t in self.tree.Nodes:
                    expr = M * self.f[t, j]
                    expr += gp.quicksum(gp.quicksum(self.l[i, s]
                                                    for s in self. _getLeftLeaves(t))
                                        for i in range(self.n) if lb <= self.data.at[i, j] <= ub)
                    num = 0
                    for i, ind in enumerate(b.t):
                        if ind == 0:
                            expr += M * self.q[t, i]
                            num = num + 1
                    expr += M * self.q[t, len(b.t)]
                    num = num + 1
                    self.model.addConstr(expr <= M + num * M)
                # right
                lb, ub = self.thresholds[j][b.lr[0]
                                            ], self.thresholds[j][b.lr[1]]
                M = np.sum(np.logical_and(
                    self.data.loc[:, j] >= lb, self.data.loc[:, j] <= ub))
                for t in self.tree.Nodes:
                    expr = M * self.f[t, j]
                    expr += gp.quicksum(gp.quicksum(self.l[i, s]
                                                    for s in self. _getRightLeaves(t))
                                        for i in range(self.n) if lb <= self.data.at[i, j] <= ub)
                    for i, ind in enumerate(b.t):
                        if ind == 1:
                            expr -= M * self.q[t, i]
                    expr -= M * self.q[t, len(b.t)]
                    self.model.addConstr(expr <= M)
            # min and max
            M = np.sum(self.thresholds[j][-1] < self.data.loc[:, j]) \
                + np.sum(self.data.loc[:, j] < self.thresholds[j][0])
            self.model.addConstrs(M * self.f[t, j]
                                  + gp.quicksum(gp.quicksum(self.l[i, s]
                                                            for s in self. _getLeftLeaves(t))
                                                for i in self.datapoints if self.thresholds[j][-1] < self.data.at[i, j])
                                  + gp.quicksum(gp.quicksum(self.l[i, s]
                                                            for s in self. _getRightLeaves(t))
                                                for i in self.datapoints if self.data.at[i, j] < self.thresholds[j][0])
                                  <= M for t in self.tree.Nodes)
        for t in self.tree.Leaves:
            for c in self.labels:
                M = np.sum(self.data.loc[:, self.label] == c)
                l_sum = 0
                for i in self.datapoints:
                    if self.data.at[i, self.label] == c:
                        l_sum += self.l[i, t]
                self.model.addConstr(l_sum - M * self.p[t, c] <= self.e[t, c])

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    def _getThresholds(self, x, y):
        """
        obtaining all possible thresholds
        """
        x = np.array(x)
        y = np.array(y)
        thresholds = []
        for j in range(len(self.cat_features)):
            threshold = []
            prev_i = np.argmin(x[:, j])
            prev_label = y[prev_i]
            for i in np.argsort(x[:, j])[1:]:
                y_cur = y[x[:, j] == x[i, j]]
                y_prev = y[x[:, j] == x[prev_i, j]]
                if (not np.all(prev_label == y_cur) or len(np.unique(y_prev)) > 1) and x[i, j] != x[prev_i, j]:
                    threshold.append((x[prev_i, j] + x[i, j]) / 2)
                    prev_label = y[i]
                prev_i = i

            # If there is only one unique value in the list then use 0.5
            # TODO DOES THIS WORK?
            if not bool(threshold):
                threshold.append(0.5)
            # threshold = [np.min(x[:, j])-1] + threshold + [np.max(x[:, j])+1]
            thresholds.append(threshold)

        return thresholds

    def _getBins(self, tmin, tmax):
        """
        obtaining the binary encoding value ranges
        """
        bin_ranges = namedtuple('Bin', ['lr', 'ur', 't'])

        if tmax <= tmin:
            return []

        if tmax - tmin <= 1:
            return [bin_ranges([tmin, tmax], [tmin, tmax], [])]

        tmid = int((tmax - tmin) / 2)
        bins = [bin_ranges([tmin, tmin+tmid+1], [tmin+tmid, tmax], [])]

        for b in self._getBins(tmin, tmin+tmid):
            bins.append(bin_ranges(b.lr, b.ur, [0] + b.t))
        for b in self._getBins(tmin+tmid+1, tmax):
            bins.append(bin_ranges(b.lr, b.ur, [1] + b.t))

        return bins

    def _getLeftLeaves(self, t):
        """
        get leaves under the left branch
        """
        tl = t * 2
        ll_index = []
        for t in self.tree.Leaves:
            tp = t
            while tp:
                if tp == tl:
                    ll_index.append(t)
                tp //= 2
        return ll_index

    def _getRightLeaves(self, t):
        """
        get leaves under the left branch
        """
        tr = t * 2 + 1
        rl_index = []
        for t in self.tree.Leaves:
            tp = t
            while tp:
                if tp == tr:
                    rl_index.append(t)
                tp //= 2
        return rl_index
