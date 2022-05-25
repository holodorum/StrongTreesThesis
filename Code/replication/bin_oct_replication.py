#!/usr/bin/python
from modelsandtree.bin_oct import BinOCT
from outputobject.output_object_bin_oct import OutputObjectbinOCT
from replication.replication import Replication
from gurobipy import *
from modelsandtree.tree import Tree
from warm_start.warmstartbin import WarmStartBin


class BinOCTReplication(Replication):
    def __init__(self, argv):
        super().__init__(argv)

    def create_model(self):
        # Tree structure: We create a tree object of depth d
        tree = Tree(self.depth)

        # We create the MIP problem by passing the required arguments
        self.primal = BinOCT(self.data_train, self.label,
                             tree, self._lambda, self.time_limit, self.mip_gap, self.mode)

    def warm_start(self):
        primal = WarmStartBin(
            self.primal, self.data_train, self.label).run_warmstart()
        return primal

    def create_output(self):
        train_len = len(self.data_train.index)

        ##########################################################
        # output setup
        ##########################################################
        outputObject = OutputObjectbinOCT('binOCT',
                                          self.input_file, self.depth, self.time_limit,  self._lambda, self.input_sample, self.calibration, self.mode, self.warmstart)
        outputObject.main(self.primal, train_len, self.solving_time, self.data_train, self.data_test, self.data_calibration,
                          print=False, final_results=True)
