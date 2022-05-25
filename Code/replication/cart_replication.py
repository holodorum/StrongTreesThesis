from replication.replication import Replication
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from outputobject.output_object_cart import OutputObjectCart
import time


class CartReplication(Replication):
    def __init__(self, argv):
        super().__init__(argv)

    def create_model(self):
        # We create the decision tree classifier by passing the required arguments
        self.clf = DecisionTreeClassifier(random_state=0, max_depth=self.depth,
                                          ccp_alpha=self._lambda)

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
        outputObject = OutputObjectCart('Cart',
                                        self.input_file, self.depth, self.time_limit, self._lambda, self.input_sample, self.calibration, self.mode)
        outputObject.main(self.clf, train_len, self.solving_time, self.data_train, self.data_test, self.label, self.data_calibration,
                          print=False, final_results=True)
