from utilities.utils import Utils


class UtilsBinOCT(Utils):
    def __init__(self):
        pass

    def get_node_status(self, grb_model, b, beta, p, n):
        '''
        This function give the status of a given node in a tree. By status we mean whether the node
            1- is pruned? i.e., we have made a prediction at one of its ancestors
            2- is a branching node? If yes, what feature do we branch on
            3- is a leaf? If yes, what is the prediction at this node?

        :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
        :param b: The values of branching decision variable b
        :param beta: The values of prediction decision variable beta
        :param p: The values of decision variable p
        :param n: A valid node index in the tree
        :return: pruned, branching, selected_feature, leaf, value

        pruned=1 iff the node is pruned
        branching = 1 iff the node branches at some feature f
        selected_feature: The feature that the node branch on
        leaf = 1 iff node n is a leaf in the tree
        value: if node n is a leaf, value represent the prediction at this node
        '''
        tree = grb_model.tree
        mode = grb_model.mode
        pruned = False
        branching = False
        leaf = False
        value = None
        selected_feature = None

        pruned = False  # We always split in binOCT

        if n in tree.Leaves:  # leaf
            leaf = True
            if mode == "regression":
                value = beta[n, 1]
            elif mode == "classification":
                for k in grb_model.labels:
                    if beta[n, k] > 0.5:
                        value = k

        if n in tree.Nodes:
            if (pruned == False) and (leaf == False):  # branching
                for f in grb_model.cat_features:
                    if b[n, f] > 0.5:
                        selected_feature = f
                        branching = True

        return pruned, branching, selected_feature, leaf, value

    def get_predicted_value(self, grb_model, local_data, b, beta, p, i):
        '''
        This function returns the predicted value for a given datapoint
        :param grb_model: The gurobi model we solved
        :param local_data: The dataset we want to compute accuracy for
        :param b: The value of decision variable b
        :param beta: The value of decision variable beta
        :param p: The value of decision variable p
        :param i: Index of the datapoint we are interested in
        :return: The predicted value for datapoint i in dataset local_data
        '''
        tree = grb_model.tree
        current = 1

        while True:
            pruned, branching, selected_feature, leaf, value = self.get_node_status(
                grb_model, b, beta, p, current)
            if leaf:
                return value
            elif branching:
                if local_data.at[i, selected_feature] == 1:  # going right on the branch
                    current = tree.get_right_children(current)
                else:  # going left on the branch
                    current = tree.get_left_children(current)
            elif pruned:
                current = tree.get_right_children(current)
