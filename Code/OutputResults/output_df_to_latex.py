from analyze_df import *
import pandas as pd


class output_df_to_latex():
    def __init__(self, output_df_real, output_df_synthetic) -> None:
        self.output_df_real = output_df_real
        self.output_df_synthetic = output_df_synthetic

    def create_pivot(self, bool_synth):
        '''Creates pivot table
        bool_synt True als synthetic dataset, false otherwise
        '''
        if bool_synth:
            df_pivot = self.output_df_synthetic.pivot_table(columns='approach', index=['depth', 'nrow', 'features'], values=[
                                                            'train_acc', 'test_acc', 'fdr', 'tdr', 'solving_time'], aggfunc=['mean', 'std'])
        else:
            df_pivot = self.output_df_real.pivot_table(columns='approach', index=[
                                                       'dataset', 'depth'], values=['train_acc', 'test_acc', 'solving_time'], aggfunc=['mean', 'std'])
        self.df_pivot = df_pivot.reindex(columns=df_pivot.columns.reindex([
            'BendersOCT', 'FlowOCT',  'binOCT', 'OCT', 'Cart'], level=2)[0]).swaplevel(0, 2, axis=1)

    def pivot_plus_minus_feature(self, feature):
        dataset_names = {'monk1_enc.csv': 'monk-1',
                         'car_evaluation_enc.csv': 'car-evaluation',
                         'balance-scale_enc.csv': 'balance-scale',
                         'kr-vs-kp_enc.csv': 'kr-vs-kp',
                         'breast-cancer_enc.csv': 'breast-cancer',
                         'soybean-small_enc.csv': 'soybean-small'}
        idx = pd.IndexSlice

        # Round the pivot
        self.df_pivot = round(self.df_pivot, 2)

        # Create DF SLICE AND ADD THE +/-
        df_sliced = self.df_pivot.loc[:, idx[[
            'BendersOCT', 'FlowOCT',  'binOCT', 'OCT', 'Cart'], feature, :]]
        # df_sliced = df_real_pivot.loc[idx[:,[2,3,4,5]], idx[['BendersOCT', 'Cart', 'FlowOCT', 'OCT', 'binOCT'], 'train_acc', : ]]
        df_sliced = df_sliced.groupby(level=[0, 1], axis=1).apply(
            lambda x: x.astype(str).apply('('.join, 1)+')')
        df_sliced = df_sliced.rename(dataset_names)
        if len(feature) == 1:
            df_sliced = df_sliced.reset_index().droplevel(1, axis=1)
        self.df_sliced = df_sliced
        return df_sliced

    def pivot_to_latex(self, bold_var=0):
        '''Create latex file from pivot table
        bold -> 0 don't make bold, 1 = make minimum bold, 2 = make maximum bold
        '''
        slice_ = ['BendersOCT', 'FlowOCT',  'binOCT', 'OCT', 'Cart']
        if bold_var == 0:
            sliced_table = self.df_sliced.style
        elif bold_var == 1:
            sliced_table = self.df_sliced.style.highlight_min(
                axis=1, props='textbf:--rwrap;', subset=slice_)
        elif bold_var == 2:
            sliced_table = self.df_sliced.style.highlight_max(
                axis=1, props='textbf:--rwrap;', subset=slice_)

        # sliced_table = sliced_table[slice_]
        # sliced_table.hide_index()
        sliced_table_latex = sliced_table.to_latex(column_format='rrrrrrr', position='ht!', position_float='centering', hrules=True,
                                                   multirow_align="t", multicol_align="r", caption='Number of observations and features of the datasets used')
        # print((df_sliced.style.highlight_max(axis = 1,props = 'textbf:--rwrap;')).to_latex(caption = 'Number of observations and features of the datasets used') )
        print(sliced_table_latex)

    def make_latex_table(self, feature, bold_var, is_synth):
        self.create_pivot(is_synth)
        self.pivot_plus_minus_feature(feature)
        self.pivot_to_latex(bold_var)


# CREATE ANALYZE CLASSES
analyze_synth = AnalyzeTimeSynthetic('test.csv')
analyze_real = AnalyzeTime('test.csv')

# LOAD DATAFRAMES
output_df = analyze_synth.load_dataset()
output_df_real, output_df_synthetic = analyze_synth.split_synthetic_and_real(
    output_df)

# FOR REAL DATASET
# Create DF With Number ofFeatures
df_results_real = analyze_real.create_results(output_df_real)
df_results_real = analyze_real.add_baseline(output_df_real, df_results_real)

# Obtain data_features
datasets = ['monk1_enc.csv', 'car_evaluation_enc.csv',
            'balance-scale_enc.csv', 'kr-vs-kp_enc.csv',
            'breast-cancer_enc.csv', 'soybean-small_enc.csv']
data_path = "c:\\Users\\bartd\\Documents\\Erasmus_\\Jaar 4\\Master Econometrie\\Thesis\\Optimal Trees\\StrongTree\\Datasets\\"

real_table_features = real_feature_table(data_path, datasets)
data_features = real_table_features.get_dataset_features()
# Add No. features
df_results_real = analyze_real.add_n_features(
    data_features)  # Add n of features

# Optional to Analyze amount of solved problems
status_df = analyze_real.n_solved(output_df)  # How many instances are solved

# Write to CSV to Analyze in R
idx = pd.IndexSlice
# df_results_real.loc[idx[['BendersOCT', 'Cart', 'FlowOCT', 'OCT', 'binOCT']],:].to_csv(
#     r'C:\Users\bartd\Documents\Erasmus_\Jaar 4\Master Econometrie\Thesis\Optimal Trees\StrongTree\Results\aresults_df_time_experiment_real.csv')

# FOR SYNTHETIC DATASET

# Load Discovery Rate and Merge
df_discovery_rate = analyze_synth.load_discovery_rate()
output_df_synthetic = analyze_synth.merge_discovery_and_synthetic(
    output_df_synthetic, df_discovery_rate)

# Create DF With Features
df_results_synth = analyze_synth.create_results(output_df_synthetic)
df_results_synth = analyze_synth.add_baseline(
    output_df_synthetic, df_results_synth)

# Obtain EncFeatureDict
obs_list = [100, 500, 2500]
features_list = [10, 30, 100]
percentage_binary = 2.0/3
latex_synth_features = synthetic_feature_table(
    obs_list, features_list, percentage_binary)
enc_feature_dict = latex_synth_features.create_feature_dict(
    features_list, percentage_binary)

# Add N of features
df_results_synth = analyze_synth.add_n_features(
    output_df_synthetic, enc_feature_dict)  # Add n of features

status_df = analyze_synth.n_solved(output_df)  # How many instances are solved

# print(df_results_synth)
# df_results_synth.loc[idx[['BendersOCT', 'Cart', 'FlowOCT', 'OCT', 'binOCT']], :].to_csv(
#     r'C:\Users\bartd\Documents\Erasmus_\Jaar 4\Master Econometrie\Thesis\Optimal Trees\StrongTree\Results\aresults_df_time_experiment_synthetic.csv')

to_latex = output_df_to_latex(output_df_real, output_df_synthetic)

# to_latex.make_latex_table(feature=['tdr', 'fdr'], bold_var=0, is_synth=True)
# to_latex.make_latex_table(feature='tdr', bold_var=2, is_synth=True)
# to_latex.make_latex_table(feature='train_acc', bold_var=2, is_synth=True)
# to_latex.make_latex_table(feature='train_acc', bold_var=2, is_synth=False)
# to_latex.make_latex_table(feature='solving_time', bold_var=0, is_synth=True)

# to_latex.make_latex_table(feature='train_acc', bold_var=2, is_synth=False)
# to_latex.make_latex_table(feature='solving_time', bold_var=0, is_synth=False)
to_latex.make_latex_table(feature='solving_time', bold_var=0, is_synth=False)
