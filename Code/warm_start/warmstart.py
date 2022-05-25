from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from collections import namedtuple


class WarmStart():
    def __init__(self, model_object, data_train, label):
        self.model_object = model_object
        self.x = data_train.loc[:, data_train.columns != label]
        self.y = data_train.loc[:, label]
        self.tree = self.model_object.tree

    # 1) Train Tree
    # 2) Get Rules
    # 3) Set Start

    def train_cart_tree(self):
        # train with CART
        # We get the depth from the tree object in the model object
        clf = DecisionTreeClassifier(max_depth=self.tree.depth)
        clf.fit(self.x, self.y)
        return clf

    def _setStart(self, rules):
        """
        set warm start from CART
        """

        # fix branch node
        for n in self.tree.Nodes:

            # split
            if rules[n].feat is not None:
                for f in self.model_object.cat_features:
                    if f == rules[n].feat:
                        self.model_object.b[n, f].start = 1
                    else:
                        self.model_object.b[n, f].start = 0
            # not split
            else:
                for f in self.model_object.cat_features:
                    self.model_object.b[n, f].start = 0

        # fix terminal nodes
        # branch node
        for n in self.tree.Nodes:
            # terminate
            if rules[n].feat is None and rules[n].value is not None:
                self.model_object.p[n].start = 1
                for k in self.model_object.labels:
                    if k == np.argmax(rules[n].value):
                        self.model_object.beta[n, k].start = 1
                    else:
                        self.model_object.beta[n, k].start = 0
            # not terminate
            else:
                self.model_object.p[n].start = 0
                for k in self.model_object.labels:
                    self.model_object.beta[n, k].start = 0
        # leaf node
        for n in self.tree.Leaves:

            # pruned
            if rules[n].value is None:
                self.model_object.p[n].start = 0
                for k in self.model_object.labels:
                    self.model_object.beta[n, k].start = 0

            # not pruned
            else:
                self.model_object.p[n].start = 1
                for k in self.model_object.labels:
                    if k == np.argmax(rules[n].value):
                        self.model_object.beta[n, k].start = 1
                    else:
                        self.model_object.beta[n, k].start = 0

    def make_nodemap(self, clf):
     # node index map
        node_map = {1: 0}
        for t in self.tree.Nodes:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r
        return node_map

    def _getRules(self, clf):
        """
        get splitting rules
        """
        clf_feature = clf.tree_.feature
        clf_threshold = clf.tree_.threshold
        clf_value = clf.tree_.value

        # NODES IN OCT and CART don't match, so we have to re-transfer
        node_map = self.make_nodemap(clf)
        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}

        # branch nodes
        for n in self.tree.Nodes:
            i = node_map[n]

            if i == -1:  # Node is non-existing in CART
                r = rule(None, None, None)
            elif clf_feature[i] == -2:
                r = rule(None, None,  clf_value[i, 0])
            else:
                feature_num = clf_feature[i]
                feature_name = self.model_object.cat_features[feature_num]
                r = rule(feature_name, clf_threshold[i], clf_value[i, 0])
            rules[n] = r

        # leave nodes
        for n in self.tree.Leaves:
            i = node_map[n]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf_value[i, 0])
            rules[n] = r
        return rules

    def run_warmstart(self):
        # train with CART
        clf = self.train_cart_tree()

        # get splitting rules
        rules = self._getRules(clf)

        self._setStart(rules)
        return self.model_object
