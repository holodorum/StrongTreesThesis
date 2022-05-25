from utilities.utils import Utils
from gurobipy import *
import os
import csv
from utilities.logger import Logger
import sys

# TODO Annotate the methods...


class OutputObject():
    def __init__(self, approach_name, input_file, depth, time_limit, _lambda, input_sample, calibration, mode, warmstart):
        '''
        :param approach_name: The method used
        :param input_file: Name of the dataset
        :param depth: Depth of the tree
        :param time_limit: The given time limit for solving the MIP
        :param _lambda: The regularization parameter in the objective
        :param input_sample: The sample used (random seed for the split)
        :param calibration: Calibration/fine-tuning or not
        :param mode: regression or classification
        '''

        self.approach_name = approach_name
        self.input_file = input_file
        self.depth = depth
        self.time_limit = time_limit
        self._lambda = _lambda
        self.input_sample = input_sample
        self.calibration = calibration
        self.mode = mode
        self.utils = Utils()
        self.warmstart = warmstart

    def set_output_location(self, final_results):
        '''
        Method that sets the name of the file and the location where
        output will be written to
        :param final_results: binary variable for whether we save in train/test or in some long filename
        '''
        self.out_put_path = os.getcwd() + '\\Results\\'
        self.out_put_name_long = self.input_file + '_' + str(self.input_sample) + '_' + self.approach_name + '_d_' + str(self.depth) + '_t_' + str(
            self.time_limit) + '_lambda_' + str(self._lambda) + '_c_' + str(self.calibration)

        if final_results and self.calibration:
            self.out_put_name = 'train'
        elif final_results:
            self.out_put_name = 'test'
        else:
            self.out_put_name = self.out_put_name_long

    def log_printed_output(self):
        '''
        Method that logs everything that is printed to a txt file
        '''
        sys.stdout = Logger(self.out_put_path +
                            self.out_put_name_long + '.txt')

    def model_parameters(self, grb_model):
        '''
        Gives the model get beta, b and p
        '''
        self.b_value = grb_model.model.getAttr("X", grb_model.b)
        self.beta_value = grb_model.model.getAttr("X", grb_model.beta)
        self.p_value = grb_model.model.getAttr("X", grb_model.p)

    def evaluation_measures(self, grb_model, data_train, data_test, data_calibration):
        '''
        For classification we report accuracy
        For regression we report MAE (Mean Absolute Error) , MSE (Mean Squared Error) and  R-squared

        over training, test and the calibration set
        '''
        self.train_acc = self.test_acc = self.calibration_acc = 0
        self.train_mae = self.test_mae = self.calibration_mae = 0
        self.train_r_2 = self.test_r_2 = self.calibration_r_2 = 0

        if self.mode == "classification":
            self.train_acc = self.utils.get_acc(
                grb_model, data_train, self.b_value, self.beta_value, self.p_value)
            self.test_acc = self.utils.get_acc(
                grb_model, data_test, self.b_value, self.beta_value, self.p_value)
            self.calibration_acc = self.utils.get_acc(
                grb_model, data_calibration, self.b_value, self.beta_value, self.p_value)
        elif self.mode == "regression":
            self.train_mae = self.utils.get_mae(
                grb_model, data_train, self.b_value, self.beta_value, self.p_value)
            self.test_mae = self.utils.get_mae(
                grb_model, data_test, self.b_value, self.beta_value, self.p_value)
            self.calibration_mae = self.utils.get_mae(
                grb_model, data_calibration, self.b_value, self.beta_value, self.p_value)

            self.train_mse = self.utils.get_mse(
                grb_model, data_train, self.b_value, self.beta_value, self.p_value)
            self.test_mse = self.utils.get_mse(
                grb_model, data_test, self.b_value, self.beta_value, self.p_value)
            self.calibration_mse = self.utils.get_mse(
                grb_model, data_calibration, self.b_value, self.beta_value, self.p_value)

            self.train_r2 = self.utils.get_r_squared(
                grb_model, data_train, self.b_value, self.beta_value, self.p_value)
            self.test_r2 = self.utils.get_r_squared(
                grb_model, data_test, self.b_value, self.beta_value, self.p_value)
            self.calibration_r2 = self.utils.get_r_squared(
                grb_model, data_calibration, self.b_value, self.beta_value, self.p_value)

    def print_eval_measures(self, grb_model):
        '''
        Prints the evaluation measures that have been computed in 
        the method evaluation_measures()
        '''
        print("obj value", grb_model.model.getAttr("ObjVal"))
        if self.mode == "classification":
            print("train acc", self.train_acc)
            print("test acc", self.test_acc)
            print("calibration acc", self.calibration_acc)
        elif self.mode == "regression":
            print("train mae", self.train_mae)
            print("train mse", self.train_mse)
            print("train r^2", self.train_r2)

    def print_tree(self, grb_model):
        '''
        This function print the derived tree with the branching features and the predictions asserted for each node
        :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
        :param b: The values of branching decision variable b
        :param beta: The values of prediction decision variable beta
        :param p: The values of decision variable p
        :return: print out the tree in the console
        '''
        tree = grb_model.tree
        for n in tree.Nodes + tree.Leaves:
            pruned, branching, selected_feature, leaf, value = self.utils.get_node_status(
                grb_model, self.b_value, self.beta_value, self.p_value, n)
            print('#########node ', n)
            if pruned:
                print("pruned")
            elif branching:
                print(selected_feature)
            elif leaf:
                print('leaf {}'.format(value))

    def print_benders_callbacks(self, grb_model, solving_time):
        print('\n\nTotal Solving Time', solving_time)
        print("obj value", grb_model.model.getAttr("ObjVal"))

        print('Total Callback counter (Integer)',
              grb_model.model._callback_counter_integer)
        print('Total Successful Callback counter (Integer)',
              grb_model.model._callback_counter_integer_success)

        print('Total Callback Time (Integer)',
              grb_model.model._total_callback_time_integer)
        print('Total Successful Callback Time (Integer)',
              grb_model.model._total_callback_time_integer_success)

    def write_results(self, grb_model, train_len, solving_time, initialized=True):
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
                        ["approach", "dataset", "nrow", "depth", "warmstart", "lambda", "time_limit",
                         "status", "obj_value", "train_acc", "gap", "node_count", "solving_time",
                         "cb_time_int", "cb_time_int_suc", "cb_counter_int", "cb_counter_int_suc",
                         "test_acc", "calib_acc", "sample"]
                    )

                # If model isn't initialized (this sometimes happens for OCT)
                # Then we return NA everywhere
                if not initialized:
                    results_writer.writerow(
                        [self.approach_name, self.input_file, train_len, self.depth, self.warmstart, self._lambda, self.time_limit,
                         grb_model.model.getAttr("Status"), grb_model.model.getAttr(
                             "ObjVal"), 'NA',
                         grb_model.model.getAttr(
                             "MIPGap") * 100, grb_model.model.getAttr("NodeCount"), solving_time,
                         grb_model.model._total_callback_time_integer, grb_model.model._total_callback_time_integer_success,
                         grb_model.model._callback_counter_integer, grb_model.model._callback_counter_integer_success,
                         'NA', 'NA', self.input_sample])
                else:
                    results_writer.writerow(
                        [self.approach_name, self.input_file, train_len, self.depth, self.warmstart, self._lambda, self.time_limit,
                         grb_model.model.getAttr("Status"), grb_model.model.getAttr(
                             "ObjVal"), self.train_acc,
                         grb_model.model.getAttr(
                             "MIPGap") * 100, grb_model.model.getAttr("NodeCount"), solving_time,
                         grb_model.model._total_callback_time_integer, grb_model.model._total_callback_time_integer_success,
                         grb_model.model._callback_counter_integer, grb_model.model._callback_counter_integer_success,
                         self.test_acc, self.calibration_acc, self.input_sample])

            elif self.mode == "regression":
                if not file_exists:  # Then Write Header
                    results_writer.writerow(
                        ["approach", "dataset", "nrow", "depth", 'warmstart', "lambda", "time_limit",
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
                     grb_model.model.getAttr("Status"),
                     grb_model.model.getAttr(
                         "ObjVal"), self.train_mae, self.train_mse, self.train_r2,
                     grb_model.model.getAttr(
                         "MIPGap") * 100, grb_model.model.getAttr("NodeCount"), solving_time,
                     grb_model.model._total_callback_time_integer, grb_model.model._total_callback_time_integer_success,
                     grb_model.model._callback_counter_integer, grb_model.model._callback_counter_integer_success,
                     self.test_mae, self.calibration_mae,
                     self.test_mse, self.calibration_mse,
                     self.test_r2, self.calibration_r2,
                     self.input_sample]
                )

    def write_model(self, grb_model):
        '''
        Method that writes the model to a file. 
        '''
        grb_model.model.write(self.out_put_path+self.out_put_name_long+'.lp')

    def main(self, grb_model, train_len, solving_time, data_train, data_test, data_calibration,
             print=False, final_results=True):
        '''
        Method that does everything from setting location,
        to logging, to evaluating and writing
        '''
        self.set_output_location(final_results)

        ##########################################################
        # Preparing the output
        ##########################################################
        try:
            self.model_parameters(grb_model)
        except gurobipy.GurobiError:
            # Call Write_results with a non-initialization boolean
            self.write_results(grb_model, train_len, solving_time, False)
            return

        ##########################################################
        # Evaluation and print the evaluation measures
        ##########################################################
        self.evaluation_measures(
            grb_model, data_train, data_test, data_calibration)

        ##########################################################
        # Print Tree, Callbacks and eval measures
        ##########################################################
        if print:
            # Using the logger method we print the output of console in a text file
            self.log_printed_output()
            self.print_tree(grb_model)
            self.print_benders_callbacks(grb_model, solving_time)
            self.print_eval_measures(grb_model)

        ##########################################################
        # writing model and results to the file
        ##########################################################
        self.write_model(grb_model)
        self.write_results(grb_model, train_len, solving_time)
