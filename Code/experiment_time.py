from replication.flow_oct_replication import FlowOCTReplication
from replication.benders_oct_replication import BendersOCTReplication
from replication.oct_replication import OCTReplication
from replication.bin_oct_replication import BinOCTReplication
from replication.cart_replication import CartReplication
from replication.cart_cv_replication import CartCVReplication
import numpy as np


def run_time_experiment():
    depths = [2, 3, 4, 5]
    # datasets = ['monk1_enc.csv', 'monk2_enc.csv', 'monk3_enc.csv', 'car_evaluation_enc.csv',
    #             'balance-scale_enc.csv', 'kr-vs-kp_enc.csv', 'house-votes-84_enc.csv', 'tic-tac-toe_enc.csv',
    #             'breast-cancer_enc.csv', 'hayes-roth_enc.csv', 'spect_enc.csv', 'soybean-small_enc.csv']
    datasets = ['monk1_enc.csv', 'car_evaluation_enc.csv',
                'balance-scale_enc.csv', 'kr-vs-kp_enc.csv',
                'breast-cancer_enc.csv', 'soybean-small_enc.csv']
    samples = [1, 2, 3, 4]
    _lambda = 0.0
    time_limit = 600
    mip_gap = 0.001
    calibration = 0

    datasets = ['balance-scale_enc.csv']
    depths = [2]

    # 6 * 4 * 4 * 4 -> 3840 minutes -> 2 2/3 dag
    for dataset in datasets:
        for depth in depths:
            for sample in samples:
                print(depth, sample)

                # cart = CartReplication(["-f", dataset, "-d", depth, "-t", time_limit, "-g", mip_gap,
                #                         "-l", 0, "-i", sample, "-c", calibration, "-m", "classification", '-w', 0])
                # cart.run()
                # bendersOCT = BendersOCTReplication(
                #     ["-n", 'BendersOCT', "-f", dataset, "-d", depth, "-t", time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', 1])
                # bendersOCT.run()
                # flowOCT = FlowOCTReplication(["-n", 'FlowOCT', "-f", dataset, "-d", depth,
                #                               "-t", time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', 1])
                # flowOCT.run()
                # binOCT = BinOCTReplication(["-n", 'BinOCT', "-f", dataset, "-d", depth, "-t",
                #                             time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', 1])
                # binOCT.run()
                oct = OCTReplication(["-n", 'OCT', "-f", dataset, "-d", depth, "-t",
                                      time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', 1])
                oct.run()

        # # create an optimal tree for every dataset
        # cartCV = CartCVReplication(["-f", dataset, "-d", 2, "-t", time_limit, "-g", mip_gap,
        #                             "-l", 0, "-i", sample, "-c", calibration, "-m", "classification", '-w', 0])
        # cartCV.run()
