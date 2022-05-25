import pandas as pd
import numpy as np
from real_feature_table import real_feature_table
from synthetic_feature_table import synthetic_feature_table


class AnalyzeTime():
    def __init__(self, filename) -> None:
        data_path = "c:\\Users\\bartd\\Documents\\Erasmus_\\Jaar 4\\Master Econometrie\\Thesis\\Optimal Trees\\StrongTree\\Results\\"
        self.location = data_path + filename

    def load_dataset(self):
        '''Load Output Dataframe'''
        output_df = pd.read_csv(self.location)
        return output_df

    def split_synthetic_and_real(self, output_df):
        # False if index is for synthetic dataset, True otherwise
        boolean_real = [
            "synthetic" not in dataset_name for dataset_name in output_df.dataset]
        output_df_real = output_df[boolean_real]
        output_df_synthetic = output_df[~np.array(boolean_real)]
        return output_df_real, output_df_synthetic

    def create_results(self, output_df):
        # Create a groupby object with the means of train_acc, solving time and mipgap
        group_means = output_df.groupby(['approach', 'dataset', 'depth'])[
            'train_acc', 'test_acc', 'solving_time', 'gap'].agg([np.mean, np.std])

        # Create a groupby object with nrow
        # TODO Add N-features
        group_rows = output_df.groupby(['approach', 'dataset', 'depth'])[
            'nrow'].agg([np.mean])

        # Combine Group Objects
        self.df_results = pd.concat([group_rows, group_means], axis=1)
        return self.df_results

    def add_baseline(self, output_df, df_results):
        # Add baseline Accuracy of CART
        group_means = output_df.groupby(['approach', 'dataset', 'depth'])[
            'train_acc', 'test_acc', 'solving_time', 'gap'].agg([np.mean, np.std])
        baseline_acc = group_means.loc['Cart', [
            ('train_acc', 'mean'), ('test_acc', 'mean')]]
        df_results = pd.merge(df_results.reset_index(), baseline_acc, left_on=[
            'dataset', 'depth'], right_index=True).groupby(['approach', 'dataset', 'depth']).sum()

        df_results.columns = ['obs_count', 'mean_train_acc', 'std_train_acc', 'mean_test_acc', 'std_test_acc',
                              'mean_solv_time', 'std_solve_time', 'mean_gap',
                              'std_gap', 'mean_train_cart', 'mean_test_cart']
        self.df_results = df_results
        return self.df_results

    def add_n_features(self, data_features):
        n_features = data_features.loc[:, 'enc_features']
        self.df_results = pd.merge(self.df_results.reset_index(), n_features, left_on=['dataset'], right_on=['Name'],
                                   right_index=True).groupby(['approach', 'dataset', 'depth']).sum()
        return self.df_results

    def n_solved(self, output_df):
        output_df.status.fillna(np.inf, inplace=True)

        # If want to sort at a specific status...
        # status_df = output_df[output_df.loc[:, 'status'] == 2, :].groupby(['approach', 'dataset', 'depth'])[
        #     'status'].agg(['value_counts'])

        status_df = output_df.groupby(['approach', 'dataset', 'depth'])[
            'status'].agg(['value_counts'])
        return status_df


class AnalyzeTimeSynthetic(AnalyzeTime):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def load_discovery_rate(self):
        '''Load Discovery Rates and Rename Approaches'''
        discovery_rate_df = pd.read_csv(
            r"C:\Users\bartd\Documents\Erasmus_\Jaar 4\Master Econometrie\Thesis\Optimal Trees\StrongTree\Results\train_discovery_rate.csv", header=0)
        approach_dict = {'Benders': 'BendersOCT',
                         'Bin': 'binOCT',
                         'Flow': 'FlowOCT',
                         }
        discovery_rate_df['approach'] = discovery_rate_df['approach'].replace(
            approach_dict)
        return discovery_rate_df

    def merge_discovery_and_synthetic(self, output_df_synthetic, discovery_rate_df):
        idx = pd.IndexSlice
        output_df_synthetic = output_df_synthetic.merge(
            discovery_rate_df.loc[:, idx['approach',
                                         'dataset', 'sample', 'tdr', 'fdr']],
            left_on=['dataset', 'sample', 'approach'], right_on=['dataset', 'sample', 'approach', ])
        return output_df_synthetic

    def add_baseline(self, output_df, df_results):
        # Add baseline Accuracy of CART
        group_means = output_df.groupby(['approach', 'dataset', 'depth'])[
            'train_acc', 'test_acc', 'solving_time', 'gap', 'fdr', 'tdr'].agg([np.mean, np.std])
        baseline_acc = group_means.loc['Cart', [
            ('train_acc', 'mean'), ('test_acc', 'mean')]]
        df_results = pd.merge(df_results.reset_index(), baseline_acc, left_on=[
            'dataset', 'depth'], right_index=True).groupby(['approach', 'dataset', 'depth']).sum()

        df_results.columns = ['obs_count', 'mean_train_acc', 'std_train_acc', 'mean_test_acc', 'std_test_acc',
                              'mean_solv_time', 'std_solve_time', 'mean_gap',
                              'std_gap', 'mean_tdr', 'std_tdr', 'mean_fdr', 'std_fdr',
                              'mean_train_cart', 'mean_test_cart']
        self.df_results = df_results
        return self.df_results

    def create_results(self, output_df):
        # Create a groupby object with the means of train_acc, solving time and mipgap
        group_means = output_df.groupby(['approach', 'dataset', 'depth'])[
            'train_acc', 'test_acc', 'solving_time', 'gap', 'fdr', 'tdr'].agg([np.mean, np.std])

        # Create a groupby object with nrow
        # TODO Add N-features
        group_rows = output_df.groupby(['approach', 'dataset', 'depth'])[
            'nrow'].agg([np.mean])

        # Combine Group Objects
        self.df_results = pd.concat([group_rows, group_means], axis=1)
        return self.df_results

    def add_n_features(self, output_df, n_enc_feature_dict):
        output_df['features'] = [int(dataset_name.split('_')[1])
                                 for dataset_name in output_df.dataset]
        output_df['enc_features'] = [
            n_enc_feature_dict[int(feature)] for feature in output_df.features]
        self.df_results = pd.merge(self.df_results.reset_index(), output_df.loc[:, ['features', 'enc_features', 'dataset']], left_on=['dataset'], right_on=['dataset'],
                                   ).groupby(['approach', 'dataset', 'depth']).mean()
        return self.df_results


# # CREATE ANALYZE CLASSES
# analyze_synth = AnalyzeTimeSynthetic('test.csv')
# analyze_real = AnalyzeTime('test.csv')

# # LOAD DATAFRAMES
# output_df = analyze_synth.load_dataset()
# output_df_real, output_df_synthetic = analyze_synth.split_synthetic_and_real(
#     output_df)

# # FOR REAL DATASET
# # Create DF With Number ofFeatures
# df_results_real = analyze_real.create_results(output_df_real)
# df_results_real = analyze_real.add_baseline(output_df_real, df_results_real)

# # Obtain data_features
# datasets = ['monk1_enc.csv', 'car_evaluation_enc.csv',
#             'balance-scale_enc.csv', 'kr-vs-kp_enc.csv',
#             'breast-cancer_enc.csv', 'soybean-small_enc.csv']
# data_path = "c:\\Users\\bartd\\Documents\\Erasmus_\\Jaar 4\\Master Econometrie\\Thesis\\Optimal Trees\\StrongTree\\Datasets\\"

# real_table_features = real_feature_table(data_path, datasets)
# data_features = real_table_features.get_dataset_features()
# # Add No. features
# df_results_real = analyze_real.add_n_features(
#     data_features)  # Add n of features

# # Optional to Analyze amount of solved problems
# status_df = analyze_real.n_solved(output_df)  # How many instances are solved

# # Write to CSV to Analyze in R
# idx = pd.IndexSlice
# # df_results_real.loc[idx[['BendersOCT', 'Cart', 'FlowOCT', 'OCT', 'binOCT']],:].to_csv(
# #     r'C:\Users\bartd\Documents\Erasmus_\Jaar 4\Master Econometrie\Thesis\Optimal Trees\StrongTree\Results\aresults_df_time_experiment_real.csv')

# # FOR SYNTHETIC DATASET

# # Load Discovery Rate and Merge
# df_discovery_rate = analyze_synth.load_discovery_rate()
# output_df_synthetic = analyze_synth.merge_discovery_and_synthetic(
#     output_df_synthetic, df_discovery_rate)

# # Create DF With Features
# df_results_synth = analyze_synth.create_results(output_df_synthetic)
# df_results_synth = analyze_synth.add_baseline(
#     output_df_synthetic, df_results_synth)

# # Obtain EncFeatureDict
# obs_list = [100, 500, 2500]
# features_list = [10, 30, 100]
# percentage_binary = 2.0/3
# latex_synth_features = synthetic_feature_table(
#     obs_list, features_list, percentage_binary)
# enc_feature_dict = latex_synth_features.create_feature_dict(
#     features_list, percentage_binary)

# # Add N of features
# df_results_synth = analyze_synth.add_n_features(
#     output_df_synthetic, enc_feature_dict)  # Add n of features

# status_df = analyze_synth.n_solved(output_df)  # How many instances are solved

# print(df_results_synth)
# # df_results.loc[idx[['BendersOCT', 'Cart', 'FlowOCT', 'OCT', 'binOCT']],:].to_csv(
# # r'C:\Users\bartd\Documents\Erasmus_\Jaar 4\Master Econometrie\Thesis\Optimal Trees\StrongTree\Results\aresults_df_time_experiment_synthetic.csv')
