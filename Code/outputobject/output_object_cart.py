from outputobject.output_object import OutputObject
from utilities.utils_oct import UtilsOCT

from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import os
import csv


class OutputObjectCart(OutputObject):
    def __init__(self, approach_name, input_file, depth, time_limit, _lambda, input_sample, calibration, mode):
        warmstart = None
        super().__init__(approach_name, input_file, depth,
                         time_limit, _lambda, input_sample, calibration, mode, warmstart)

    def evaluation_measures(self, tree_clf, data_train, data_test, data_calibration, label):
        x_train = data_train.loc[:, data_train.columns != label]
        y_train = data_train.loc[:, label]
        x_test = data_test.loc[:, data_test.columns != label]
        y_test = data_test.loc[:, label]
        x_calib = data_calibration.loc[:,
                                       data_calibration.columns != label]
        y_calib = data_calibration.loc[:, label]

        '''
        For classification we report accuracy
        For regression we report MAE (Mean Absolute Error) , MSE (Mean Squared Error) and  R-squared

        over training, test and the calibration set
        '''
        self.train_acc = self.test_acc = self.calibration_acc = 0
        self.train_mae = self.test_mae = self.calibration_mae = 0
        self.train_r_2 = self.test_r_2 = self.calibration_r_2 = 0

        if self.mode == "classification":
            self.train_acc = accuracy_score(y_train, tree_clf.predict(x_train))
            self.test_acc = accuracy_score(y_test, tree_clf.predict(x_test))
            self.calibration_acc = accuracy_score(
                y_calib, tree_clf.predict(x_calib))
        elif self.mode == "regression":
            self.train_mae = mean_absolute_error(
                y_train, tree_clf.predict(x_train))
            self.test_mae = mean_absolute_error(
                y_test, tree_clf.predict(x_test))
            self.calibration_mae = mean_absolute_error(
                y_calib, tree_clf.predict(x_calib))

            self.train_mse = mean_squared_error(
                y_train, tree_clf.predict(x_train))
            self.test_mse = mean_squared_error(
                y_test, tree_clf.predict(x_test))
            self.calibration_mse = mean_squared_error(
                y_calib, tree_clf.predict(x_calib))

            self.train_r2 = r2_score(y_train, tree_clf.predict(x_train))
            self.test_r2 = r2_score(y_test, tree_clf.predict(x_test))
            self.calibration_r2 = r2_score(y_calib, tree_clf.predict(x_calib))

    def print_eval_measures(self, grb_model):
        '''
        Prints the evaluation measures that have been computed in 
        the method evaluation_measures()
        '''
        if self.mode == "classification":
            print("train acc", self.train_acc)
            print("test acc", self.test_acc)
            print("calibration acc", self.calibration_acc)
        elif self.mode == "regression":
            print("train mae", self.train_mae)
            print("train mse", self.train_mse)
            print("train r^2", self.train_r2)

    def print_tree(self, tree_clf, feature_names):
        '''
        This function print the derived tree with the branching features and the predictions asserted for each node
        :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
        :param b: The values of branching decision variable b
        :param beta: The values of prediction decision variable beta
        :param p: The values of decision variable p
        :return: print out the tree in the console
        '''
        n_nodes = tree_clf.tree_.node_count
        children_left = tree_clf.tree_.children_left
        children_right = tree_clf.tree_.children_right
        feature = tree_clf.tree_.feature
        threshold = tree_clf.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        for n in range(n_nodes):
            print('#########node ', n)
            if is_leaves[n]:
                print('leaf {}'.format(n))
            else:
                print(feature_names[feature[n]])
        print()

    def write_results(self, grb_model, train_len, solving_time):
        '''
        Method that writes the results to the output file. 
        :param train_len: length of the training data
        :param solving_time: time it took the solver to optimize the model.
        '''

        # writing info to the file
        result_file = self.out_put_name + '.csv'

        file_exists = os.path.isfile(self.out_put_path + result_file)

        with open(self.out_put_path + result_file, mode='a', newline='') as results:
            results_writer = csv.writer(
                results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

            if self.mode == "classification":
                if not file_exists:  # Then Write Header
                    results_writer.writerow(
                        ["approach", "dataset", "nrow", "depth", "warmstart",  "lambda", "time_limit",
                         "status", "obj_value", "train_acc", "gap", "node_count", "solving_time",
                         "cb_time_int", "cb_time_int_suc", "cb_counter_int", "cb_counter_int_suc",
                         "test_acc", "calib_acc", "sample"]
                    )

                results_writer.writerow(
                    [self.approach_name, self.input_file, train_len, self.depth, self.warmstart, self._lambda, self.time_limit,
                     'NA', 'NA', self.train_acc,
                     'NA', 'NA', solving_time,
                     'NA', 'NA',
                     'NA', 'NA',
                     self.test_acc, self.calibration_acc, self.input_sample])
            elif self.mode == "regression":
                if not file_exists:  # Then Write Header
                    results_writer.writerow(
                        ["approach", "dataset", "nrow", "depth", "warmstart", "lambda", "time_limit",
                         "status", "obj_value", "train_mae", "train_mse", "train_r2",
                         "gap", "node_count", "solving_time",
                         "cb_time_int", "cb_time_int_suc", "cb_counter_int", "cb_counter_int_suc",
                         "test_mae", "calib_mae",
                         "test_mse", "calib_mse",
                         "test_r2", "calib_r2",
                         "sample"]
                    )
                results_writer.writerow(
                    [self.approach_name, self.input_file, train_len, self.depth, self.warmstart, self._lambda, self.time_limit,
                     'NA', 'NA', self.train_mae, self.train_mse, self.train_r2,
                     'NA', 'NA', solving_time,
                     'NA', 'NA',
                     'NA', 'NA',
                     self.test_mae, self.calibration_mae,
                     self.test_mse, self.calibration_mse,
                     self.test_r2, self.calibration_r2,
                     self.input_sample]
                )

    def main(self, tree_clf, train_len, solving_time, data_train, data_test, label, data_calibration,
             print=False, final_results=True):
        '''
        Method that does everything from setting location,
        to logging, to evaluating and writing
        '''
        self.set_output_location(final_results)

        ##########################################################
        # Evaluation and print the evaluation measures
        ##########################################################
        self.evaluation_measures(
            tree_clf, data_train, data_test, data_calibration, label)

        ##########################################################
        # Print Tree, Callbacks and eval measures
        ##########################################################
        if print:
            # Using the logger method we print the output of console in a text file
            self.log_printed_output()
            self.print_tree(tree_clf, data_train.columns)
            self.print_eval_measures(tree_clf)

        ##########################################################
        # writing model and results to the file
        ##########################################################
        self.write_results(tree_clf, train_len, solving_time)
