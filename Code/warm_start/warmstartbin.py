from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from collections import namedtuple
from warm_start.warmstart import WarmStart


class WarmStartBin(WarmStart):
    def __init__(self, model_object, data_train, label):
        super().__init__(model_object, data_train, label)

    def _setStart(self, rules):
        """
        set warm start from CART
        """

        # Branch Nodes
        for n in self.tree.Nodes:
            # split
            if rules[n].feat is not None:
                for feat in self.model_object.cat_features:
                    if feat == rules[n].feat:
                        self.model_object.f[n, feat].start = 1
                    else:
                        self.model_object.f[n, feat].start = 0
            # Not split (Terminal node/pruned node), not possible in bin
            else:
                pass

        # leave nodes
        for n in self.tree.Leaves:
            # Termination Node
            if rules[n].feat is None and rules[n].value is not None:
                for k in self.model_object.labels:
                    if k == np.argmax(rules[n].value):
                        self.model_object.p[n, k].start = 1
                    else:
                        self.model_object.p[n, k].start = 0
            # Pruned Node, But nothing to set there
            else:
                pass
