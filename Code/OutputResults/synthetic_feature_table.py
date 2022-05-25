import numpy as np
import pandas as pd


class synthetic_feature_table():
    def __init__(self, obs_list, features_list, percentage_binary) -> None:
        self.obs_list = obs_list
        self.features_list = features_list
        self.percentage_binary = percentage_binary

    def create_feature_dict(self, features_list, percentage_binary):
        '''
        Calculates the number of features we end up with after encoding.
        '''
        n_enc_feature_dict = {}
        for n_feature in features_list:
            n_encoded_features = n_feature - \
                np.floor(n_feature*percentage_binary)
            n_enc_feature_dict[n_feature] = int(
                (n_feature-n_encoded_features) + n_encoded_features * 3)

        return n_enc_feature_dict

    def create_table_synth(self, obs_list, features_list, feature_enc_dict):
        '''
        Create DataFrame that will be printed to latex
        '''
        data_list = []
        for obs in obs_list:
            for features in features_list:
                data_list.append([obs, features, feature_enc_dict[features]])
        synthetic_df = pd.DataFrame(
            data_list, columns=['Observations', 'Features', 'Features Encoded'])
        return synthetic_df

    def table_to_latex(self, synth_feature_df):
        '''Print table to latex'''
        print(synth_feature_df.to_latex(bold_rows=True, index=False,
              caption='Number of observations and features of the synthetic dataset'))

    def main(self):
        '''Run all methods'''
        enc_feature_dict = self.create_feature_dict(
            self.features_list, self.percentage_binary)
        synthetic_df = self.create_table_synth(
            self.obs_list, self.features_list, enc_feature_dict)
        self.table_to_latex(synthetic_df)


# obs_list = [100, 500, 2500]
# features_list = [10, 30, 100]
# percentage_binary = 2.0/3
# latex_synth_features =synthetic_feature_table(
#     obs_list, features_list, percentage_binary)
# latex_synth_features.main()
