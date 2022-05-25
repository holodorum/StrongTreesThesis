from numpy import real
import pandas as pd


class real_feature_table():
    def __init__(self, path, datasets) -> None:
        self.datasets = datasets
        self.data_path = path

    def get_dataset_features(self):
        '''Iterates through all datasets and gets dataset features'''

        #Save in DataLIst
        cols = ['Name', 'Observations', 'enc_features', 'features']
        data_list = []

        for dataset in self.datasets:
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

            list_row = [dataset, data_enc.shape[0],
                        data_enc.shape[1], data.shape[1]]
            data_list.append(list_row)

        # Make DataFrame
        data_features = pd.DataFrame(data=data_list, columns=cols)
        data_features.set_index('Name', inplace=True)
        data_features.sort_values(
            by="Observations", inplace=True,  ascending=False)
        data_features = data_features.reindex(
            columns=["Observations", 'features', 'enc_features'])

        return data_features

    def df_to_latex(self, data_features, dataset_names):
        print(data_features.rename(index=dataset_names).to_latex(
            bold_rows=True, caption='Number of observations and features of the datasets used'))


datasets = ['monk1_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv',
            'breast-cancer_enc.csv', 'soybean-small_enc.csv']


dataset_names = {'monk1_enc.csv': 'monk-1',
                 'car_evaluation_enc.csv': 'car-evaluation',
                 'balance-scale_enc.csv': 'balance-scale',
                 'kr-vs-kp_enc.csv': 'kr-vs-kp',
                 'breast-cancer_enc.csv': 'breast-cancer',
                 'soybean-small_enc.csv': 'soybean-small'}

data_path = "c:\\Users\\bartd\\Documents\\Erasmus_\\Jaar 4\\Master Econometrie\\Thesis\\Optimal Trees\\StrongTree\\Datasets\\"

real_table_features = real_feature_table(data_path, datasets)
data_features = real_table_features.get_dataset_features()
real_table_features.df_to_latex(data_features, dataset_names)
