#!/usr/bin/python
from outputobject.output_object import OutputObject
from gurobipy import *
import time
from modelsandtree.tree import Tree
from modelsandtree.benders_oct import BendersOCT
from modelsandtree.benders_oct_callback import mycallback
from replication.replication import Replication
from warm_start.warmstart import WarmStart


class BendersOCTReplication(Replication):
    def __init__(self, argv):
        super().__init__(argv)

    def create_model(self):
        # Tree structure: We create a tree object of depth d
        tree = Tree(self.depth)

        # We create the MIP problem by passing the required arguments
        # Note that I call it the primal while it actually is the master problem.
        self.master = BendersOCT(
            self.data_train, self.label, tree, self._lambda, self.time_limit,  self.mip_gap, self.mode)

    def run_model(self):
        ##########################################################
        # Creating and Solving the problem
        ##########################################################
        # We create the master problem by passing the required arguments
        start_time = time.time()

        self.create_model()

        self.master.create_master_problem()

        if self.warmstart:
            self.master = self.warm_start()

        self.master.model.update()
        self.master.model.optimize(mycallback)
        end_time = time.time()
        self.solving_time = end_time - start_time

    def warm_start(self):
        master = WarmStart(
            self.master, self.data_train, self.label).run_warmstart()
        return master

    def create_output(self):
        train_len = len(self.data_train.index)

        ##########################################################
        # output setup
        ##########################################################
        outputObject = OutputObject('BendersOCT',
                                    self.input_file, self.depth, self.time_limit, self._lambda, self.input_sample, self.calibration, self.mode, self.warmstart)
        outputObject.main(self.master, train_len, self.solving_time, self.data_train, self.data_test, self.data_calibration,
                          print=False, final_results=True)
