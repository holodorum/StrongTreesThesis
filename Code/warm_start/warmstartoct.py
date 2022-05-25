import numpy as np
from collections import namedtuple
from warm_start.warmstart import WarmStart


class WarmStartOCT(WarmStart):
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
                self.model_object.p[n].start = 1
                for feat in self.model_object.cat_features:
                    if feat == rules[n].feat:
                        self.model_object.b[feat, n].start = 1
                    else:
                        self.model_object.b[feat, n].start = 0
            # Not split (Terminal node/pruned node), not possible in bin
            else:
                self.model_object.p[n].start = 0
                for feat in self.model_object.cat_features:
                    self.model_object.b[feat, n].start = 0

                # Early termination
                if rules[n].value is not None:
                    for _ in range(100):
                        n = n*2+1
                        if n in self.tree.Leaves:
                            break

                    for k in self.model_object.labels:
                        if k == np.argmax(rules[n].value):
                            self.model_object.beta[k, n].start = 1
                        else:
                            self.model_object.beta[k, n].start = 0

        for n in self.tree.Leaves:
            if rules[n].value is None:
                pass
            else:
                for k in self.model_object.labels:
                    if k == np.argmax(rules[n].value):
                        self.model_object.beta[k, n].start = 1
                    else:
                        self.model_object.beta[k, n].start = 0

        # leave nodes
        for n in self.tree.Leaves:
            # Termination Node
            if rules[n].feat is None and rules[n].value is not None:
                for k in self.model_object.labels:
                    if k == np.argmax(rules[n].value):
                        self.model_object.beta[k, n].start = 1
                    else:
                        self.model_object.beta[k, n].start = 0
            # Pruned Node, But nothing to set there
            else:
                pass
