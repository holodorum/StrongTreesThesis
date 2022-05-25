#!/usr/bin/python
from outputobject.output_object import OutputObject
from gurobipy import *
import pandas as pd
import os
import sys
import time
from modelsandtree.tree import Tree
from modelsandtree.flow_oct import FlowOCT
from warm_start.warmstart import WarmStart

import getopt

from sklearn.model_selection import train_test_split


import abc


class Replication():
    '''
    Depending on the value of input_sample we choose one of the following random seeds and then split the whole data
    into train, test and calibration
    '''
    random_states_list = [41, 23, 45, 36, 19, 123]

    def __init__(self, argv, label='target'):
        self.set_init_args(argv)
        '''Name of the column in the dataset representing the class label.
        In the datasets we have, we assume the label is target. Please change this value at your need'''
        self.label = label

    def set_init_args(self, argv):
        try:
            opts, args = getopt.getopt(argv, "n:f:d:t:g:l:i:c:m:w:",
                                       ["method_name", "input_file=", "depth=", "timelimit=", "mipgap", "lambda=",
                                        "input_sample=",
                                        "calibration=", "mode=", "warmstart="])
        except getopt.GetoptError:
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-n", "--method_name"):
                self.method_name = arg
            if opt in ("-f", "--input_file"):
                self.input_file = arg
            elif opt in ("-d", "--depth"):
                self.depth = int(arg)
            elif opt in ("-t", "--timelimit"):
                self.time_limit = int(arg)
            elif opt in ("-g", "--mipgap"):
                self.mip_gap = float(arg)
            elif opt in ("-l", "--lambda"):
                self._lambda = float(arg)
            elif opt in ("-i", "--input_sample"):
                self.input_sample = int(arg)
            elif opt in ("-c", "--calibration"):
                self.calibration = int(arg)
            elif opt in ("-m", "--mode"):
                self.mode = arg
            elif opt in ("-w", "--warmstart"):
                self.warmstart = arg

    def load_data(self):
        data_path = os.getcwd() + '\\DataSets\\'
        self.data = pd.read_csv(data_path + self.input_file)

    def split_data(self):
        ##########################################################
        # data splitting
        ##########################################################
        '''
        Creating  train, test and calibration datasets
        We take 50% of the whole data as training, 25% as test and 25% as calibration

        When we want to calibrate _lambda, for a given value of _lambda we train the model on train and evaluate
        the accuracy on calibration set and at the end we pick the _lambda with the highest accuracy.

        When we got the calibrated _lambda, we train the mode on (train+calibration) which we refer to it as
        data_train_calibration and evaluate the accuracy on (test)
        '''
        data_train, data_test = train_test_split(
            self.data, test_size=0.25, random_state=Replication.random_states_list[self.input_sample - 1])
        data_train_calibration, data_calibration = train_test_split(data_train, test_size=0.33,
                                                                    random_state=Replication.random_states_list[self.input_sample - 1])

        if self.calibration == 1:  # in this mode, we train on 50% of the data; otherwise we train on 75% of the data
            data_train = data_train_calibration

        self.data_train = data_train
        self.data_test = data_test
        self.data_train_calibration = data_train_calibration
        self.data_calibration = data_calibration

    @abc.abstractmethod  # TODO WHY NO WARNING WHEN THIS METHOD ISN"T IMPLEMENTED?
    def create_model(self):
        # Tree structure: We create a tree object of depth d
        tree = Tree(self.depth)

        # We create the MIP problem by passing the required arguments
        self.primal = FlowOCT(self.data_train, self.label,
                              tree, self._lambda, self.time_limit, self.mode)

    @abc.abstractmethod
    def warm_start(self):
        primal = WarmStart(
            self.primal, self.data_train, self.label).run_warmstart()
        return primal

    def run_model(self):
        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        start_time = time.time()

        self.create_model()

        self.primal.create_primal_problem()

        if self.warmstart:
            self.primal = self.warm_start()

        self.primal.model.update()
        self.primal.model.optimize()
        end_time = time.time()
        self.solving_time = end_time - start_time

    @abc.abstractmethod  # TODO WHY NO WARNING WHEN THIS METHOD ISN"T IMPLEMENTED?
    def create_output(self):
        train_len = len(self.data_train.index)

        ##########################################################
        # output setup
        ##########################################################
        outputObject = OutputObject(self.method_name,
                                    self.input_file, self.depth, self.time_limit, self._lambda, self.input_sample, self.calibration, self.mode, self.warmstart)
        outputObject.main(self.primal, train_len, self.solving_time, self.data_train, self.data_test, self.data_calibration,
                          print=True, final_results=True)

    def run(self):
        self.load_data()
        self.split_data()
        self.run_model()
        self.create_output()


if __name__ == "__main__":
    main(sys.argv[1:])
