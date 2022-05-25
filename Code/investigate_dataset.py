import pandas as pd
import os
import numpy as np

datasets = ['monk1_enc.csv', 'monk2_enc.csv', 'monk3_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv', 'house-votes-84_enc.csv', 'tic-tac-toe_enc.csv',
            'breast-cancer_enc.csv', 'hayes-roth_enc.csv', 'spect_enc.csv', 'soybean-small_enc.csv']

data_path = os.getcwd() + '\\DataSets\\'
dataset_size = pd.DataFrame({'name': [],
                            'observations': [],
                             'enc_features': [],
                             'features': []}
                            )
cols = ['Name', 'Observations', 'enc_features', 'features']
data_list = []
for dataset in datasets:
    data_enc = pd.read_csv(data_path + dataset)

    name_non_enc = dataset.split('_enc')[0]
    if name_non_enc in ['monk1', 'monk2', 'monk3', 'car_evaluation']:
        data = pd.read_csv(data_path+name_non_enc+'.csv', sep=';')
    else:
        # TODO Preferably do this without a double try block
        try:
            data = pd.read_csv(data_path + name_non_enc+'.data')
        except:
            try:
                data = pd.read_csv(data_path+name_non_enc+'.csv')
            except:
                data = pd.read_csv(data_path+name_non_enc+'.train')

    list_row = [name_non_enc, data_enc.shape[0],
                data_enc.shape[1], data.shape[1]]
    data_list.append(list_row)

data_features = pd.DataFrame(data=data_list, columns=cols)
print(data_features)
