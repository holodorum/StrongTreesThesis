from pyexpat import features
import pandas as pd
import os
import numpy as np
import csv
from syntethicexperiment.generate_data import GroundtruthDataset
from replication.flow_oct_replication import FlowOCTReplication
from replication.flow_oct_replication import FlowOCTReplication
from replication.benders_oct_replication import BendersOCTReplication
from replication.oct_replication import OCTReplication
from replication.bin_oct_replication import BinOCTReplication
from replication.cart_replication import CartReplication
from replication.cart_cv_replication import CartCVReplication
from syntethicexperiment.utils import tree_features, discovery_rate


def run_methods(dataset_name, depth_tree, time_limit, mip_gap, _lambda, sample, calibration, warmstart, grnd_features):
    features_used_dict = {}

    print('Benders')
    bendersOCT = BendersOCTReplication(
        ["-n", 'BendersOCT', "-f", dataset_name, "-d", depth_tree, "-t", time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', warmstart])
    bendersOCT.run()
    features_used_dict['benders'] = tree_features.features_used(
        bendersOCT.master)

    print('Cart')
    cart = CartReplication(["-n", 'Cart', "-f", dataset_name, "-d", depth_tree, "-t", time_limit,
                            "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', 0])
    cart.run()
    feature_index = tree_features.features_used_cart(cart.clf)
    features_used_dict['cart'] = cart.data.columns[feature_index]

    print('Flow')
    flowOCT = FlowOCTReplication(["-n", 'Flow', "-f", dataset_name, "-d", depth_tree,
                                  "-t", time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', warmstart])
    flowOCT.run()
    features_used_dict['flow'] = tree_features.features_used(flowOCT.primal)

    print('Bin')
    binOCT = BinOCTReplication(["-n", "binOCT", "-f", dataset_name, "-d", depth_tree, "-t",
                                time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', warmstart])
    binOCT.run()
    features_used_dict['bin'] = tree_features.features_used_bin(binOCT.primal)

    print('OCT')
    oct = OCTReplication(["-n", "OCT", "-f", dataset_name, "-d", depth_tree, "-t",
                          time_limit, "-g", mip_gap, "-l", _lambda, "-i", sample, "-c", calibration, "-m", "classification", '-w', warmstart])
    oct.run()
    features_used_dict['oct'] = tree_features.features_used_oct(oct.primal)

    list_tdr, list_fdr = compute_discovery_rate(
        grnd_features, features_used_dict)
    return list_tdr, list_fdr


def compute_discovery_rate(grnd_features, feature_dict):
    list_tdr = []
    list_fdr = []
    for key in feature_dict:
        TDR, FDR = discovery_rate.discovery_rate(
            grnd_features, feature_dict[key])

        list_tdr.append(TDR)
        list_fdr.append(FDR)
    return list_tdr, list_fdr


def run_synthetic_experiment():
    # Dataset Settings
    obs_list = [100, 500, 2500]
    features_list = [10, 30, 100]
    percentage_binary = 2.0/3
    depth = [2, 3, 4, 5]

    # Method Settings
    time_limit = 600
    mip_gap = 0.001
    _lambda = 0.0
    calibration = 0
    warmstart = 1
    samples = [1, 2, 3, 4]

    # # test_run
    # obs_list = [100]
    # features_list = [10]
    # percentage_binary = 2.0/3
    # depth = [3]
    # samples = [3]

    fd = open('Results\\train_discovery_rate.csv', 'a', newline='')
    results_writer = csv.writer(
        fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    results_writer.writerow(["approach", "dataset", 'tdr', 'fdr', 'sample'])
    fd.close()

    # 4*3*4*4*4 -> 7680 minuten -> 5... days
    # Started Yesterday Around 11. Almost 1/3 After Less than 12 hours.
    for n_obs in obs_list:
        for n_features in features_list:
            for depth_tree in depth:
                if n_features in [10, 30]:
                    continue
                if n_obs in [100, 500]:
                    continue
                if depth_tree in [2, 3]:
                    continue

                # CREATE DATASET
                grnd_dataset, grnd_features = GroundtruthDataset.get_dataset_and_features(
                    n_obs, n_features, percentage_binary, depth_tree)

                # WRITE DATASET
                dataset_name = "synthetic" + \
                    str(n_obs) + '_' + str(n_features) + \
                    "_" + str(depth_tree) + '.csv'
                grnd_dataset.to_csv(
                    os.getcwd() + '\\DataSets\\' + dataset_name, index=False)

                for sample in samples:
                    print(n_features, depth_tree, n_obs, sample)
                    if depth_tree == 4 and sample in [1, 2]:
                        continue
                    list_tdr, list_fdr = run_methods(dataset_name, depth_tree, time_limit,
                                                     mip_gap, _lambda, sample, calibration, warmstart, grnd_features)

                    # NOT GOOD PRACTICE, BUT QUICK FIX, THIS IS ORDER OF THE USED METHODS
                    dataset_name_list = [dataset_name for i in range(5)]
                    sample_list = [sample for i in range(5)]
                    method_names = ['Benders', 'Cart', 'Flow', 'Bin', 'OCT']

                    discovery_df = pd.DataFrame(
                        [method_names, dataset_name_list, list_tdr, list_fdr, sample_list]).T
                    discovery_df.to_csv(
                        'Results\\train_discovery_rate.csv', mode='a', index=False, header=False)
