from replication.replication import Replication
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from outputobject.output_object_cart import OutputObjectCart
import numpy as np

import time


class CartCVReplication(Replication):
    def __init__(self, argv):
        super().__init__(argv)

    def randomized_search(self):
        x = self.data_train.loc[:, self.data_train.columns != self.label]
        y = self.data_train.loc[:, self.label]

        max_features = ['auto', 'log2', None]
        max_depth = [int(x) for x in np.linspace(1, 15)]
        min_samples_split = [2, 6, 10]
        min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=2)]
        max_leaf_nodes = [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        parameters = {'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'max_leaf_nodes': max_leaf_nodes
                      }

        # If there is a dataset with less than 5 cases for a class we cannot do cv with 5.
        cv_num = np.min([y.value_counts().min(), 5])

        tuned_clf = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), param_distributions=parameters, n_iter=1000,
                                       cv=cv_num, random_state=0)

        tuned_clf.fit(x, y)
        return tuned_clf.best_estimator_

    def create_model(self):
        # We create the decision tree classifier by passing the required arguments
        self.clf = self.randomized_search()

    def run_model(self):
        x = self.data_train.loc[:, self.data_train.columns != self.label]
        y = self.data_train.loc[:, self.label]

        self.create_model()

        start_time = time.time()
        self.clf.fit(x, y)
        end_time = time.time()
        self.solving_time = end_time - start_time

    def create_output(self):
        train_len = len(self.data_train.index)

        ##########################################################
        # output setup
        ##########################################################
        outputObject = OutputObjectCart('CartCV',
                                        self.input_file, self.clf.get_depth(), self.time_limit, self._lambda, self.input_sample, self.calibration, self.mode)
        outputObject.main(self.clf, train_len, self.solving_time, self.data_train, self.data_test, self.label, self.data_calibration,
                          print=False, final_results=True)
