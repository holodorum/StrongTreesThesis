'''
This module formulate the OCT problem in gurobipy.
'''
from gurobipy import *
import numpy as np
import pandas as pd
from scipy import stats


class OCT:
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

        # Gurobi model
        self.model = Model('OCT')
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
        This function create and return a gurobi model formulating the OCT problem
        :return:  gurobi model object with the OCT formulation
        '''
        # model sense
        self.model.modelSense = GRB.MAXIMIZE

        # define variables

        # Number of missclassified samples
        self.L = self.model.addVars(
            self.tree.Leaves, vtype=GRB.INTEGER, name='L')
        # M[k,n] = number of samples with class k at leave node n
        self.M = self.model.addVars(
            self.labels, self.tree.Leaves, vtype=GRB.INTEGER, name='M')
        # N[n] = number of samples at leave node n
        self.N = self.model.addVars(
            self.tree.Leaves, vtype=GRB.INTEGER, name='N')
        # #l[n] = 1 when number of samples at node n is more than S_min (to prevent overfitting)
        self.l = self.model.addVars(
            self.tree.Leaves, vtype=GRB.BINARY, name='l')
        # h[i,n] = 1 when datapoint i is assigned to leave node n
        self.h = self.model.addVars(
            self.datapoints, self.tree.Leaves, vtype=GRB.BINARY, name='z')
        # b[f,n] = 1 when we branch on feature f on branching node n
        self.b = self.model.addVars(
            self.cat_features, self.tree.Nodes, vtype=GRB.BINARY, name='b')
        '''
        For classification beta[n,k]=1 iff at node n we predict class k
        For the case regression beta[n,1] is the prediction value for node n
        '''
        self.beta = self.model.addVars(
            self.labels, self.tree.Leaves, vtype=GRB.BINARY, name='beta')
        # p[n] = 1 if we decide to split on node n
        self.p = self.model.addVars(
            self.tree.Nodes, vtype=GRB.BINARY, name='p')
        # dt[n] the decision treshold that is used at node n
        self.dt = self.model.addVars(
            self.tree.Nodes, vtype=GRB.CONTINUOUS, name='dt')

        ############################### define constraints#################################
        # L[n] >= N[n] - M[n,k] - |I|*(1-beta[n,k]) forall n in leaves and k in classes
        self.model.addConstrs((self.L[n] >= self.N[n] - self.M[k, n] - len(self.datapoints)*(
            1-self.beta[k, n])) for n in self.tree.Leaves for k in self.labels)

        # L[n] <= N[n] - M[n,k] + |I|*(beta[n,k]) forall n in leaves and k in classes
        self.model.addConstrs((self.L[n] <= self.N[n] - self.M[k, n] + len(
            self.datapoints)*self.beta[k, n]) for n in self.tree.Leaves for k in self.labels)

        # M[n,k] = sum(h[i,n], for all i where y_i = k) forall n in leaves and k in classes
        self.model.addConstrs((quicksum(self.h[i, n] for i in self.datapoints if (self.data.at[i, self.label] == k)) ==
                               self.M[k, n]) for n in self.tree.Leaves for k in self.labels)

        # N[n] = sum(h[i,n], forall i) forall n in leaves
        self.model.addConstrs((quicksum(
            self.h[i, n] for i in self.datapoints) == self.N[n]) for n in self.tree.Leaves)

        # sum(beta[n,k], k in labels) = l[n] for n in leaves
        self.model.addConstrs((quicksum(
            self.beta[k, n] for k in self.labels) == self.l[n]) for n in self.tree.Leaves)

        # h[i,n] <= l[n] for n in leaves
        self.model.addConstrs((self.h[i, n] <= self.l[n])
                              for n in self.tree.Leaves for i in self.datapoints)

        # sum(h[i,n], for n in leaves) = 1 forall i
        self.model.addConstrs(
            (quicksum(self.h[i, n] for n in self.tree.Leaves) == 1) for i in self.datapoints)

        # sum((b[f,s]) for f in features if f = 0) >= 2h[i,n] - 1 forall i, n in branches, s in left branch ancestors
        # sum((b[f,s]) for f in features if f = 1) >= h[i,n] forall i, n in branches, s in righ branch ancestors
        # calculate minimum distance
        min_dis = self._calMinDist(
            self.data.iloc[:, self.data.columns != self.label])
        for n in self.tree.Leaves:
            left = (n % 2 == 0)
            ancestor = n // 2
            while ancestor != 0:
                if left:
                    self.model.addConstrs((quicksum(self.b[f, ancestor] for f in self.cat_features if self.data.at[i, f] == 0)
                                           >=
                                           2*self.h[i, n] - 1) for i in self.datapoints)
                else:
                    self.model.addConstrs((quicksum((self.b[f, ancestor]) for f in self.cat_features if self.data.at[i, f] == 1)
                                           >=
                                           self.h[i, n]) for i in self.datapoints)
                left = (ancestor % 2 == 0)
                ancestor //= 2

        # sum(b[f,n], f in features) == p[n] for all n in Nodes
        self.model.addConstrs((quicksum(
            self.b[f, n] for f in self.cat_features) == self.p[n]) for n in self.tree.Nodes)

        # 0<=dt[n]<=d[n] for all n in Nodes
        # TODO larger than 0?
        self.model.addConstrs((self.dt[n] <= self.p[n])
                              for n in self.tree.Nodes)
        # self.model.addConstrs((self.dt[n] >= 0) for n in self.tree.Nodes)
        # d[n] <= d[p(n)] for all n in Nodes except for node 1
        for n in self.tree.Nodes:
            if n != 1:
                n_parent = self.tree.get_parent(n)
                self.model.addConstr((self.p[n] <= self.p[n_parent]))

        # define objective function
        # ((1-lambda)/baseline) sum(L, n in leaves) + lambda sum(d[n], n in branch)
        baseline = self._calBaseline(self.data.loc[:, self.label])
        baseline = 1
        obj = LinExpr(0)
        for n in self.tree.Leaves:
            obj.add((1-self._lambda)*(len(self.datapoints)-self.L[n]))

        for n in self.tree.Nodes:
            obj.add(-1*self._lambda*self.p[n])

        self.model.setObjective(obj)

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _calMinDist(x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x.iloc[:, j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return pd.Series(min_dis, index=x.columns)
