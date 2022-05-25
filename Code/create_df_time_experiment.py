import pandas as pd
import numpy as np
import os


class AnalyzeTime():
    def __init__(self, filename) -> None:
        self.location = os.getcwd() + '\\Results\\' + filename

    def load_dataset(self):
        output_df = pd.read_csv(self.location)
        return output_df

    def create_results(self, output_df):
        # Create a groupby object with the means of train_acc, solving time and mipgap
        group_means = output_df.groupby(['approach', 'dataset', 'depth'])[
            'train_acc', 'solving_time', 'gap'].agg([np.mean, np.std])

        # Create a groupby object with nrow
        # TODO Add N-features
        group_rows = output_df.groupby(['approach', 'dataset', 'depth'])[
            'nrow'].agg([np.mean])

        # Combine Group Objects
        df_results = pd.concat([group_rows, group_means], axis=1)

        # Add baseline Accuracy of CART
        baseline_acc = group_means.loc['Cart', [('train_acc', 'mean')]]
        df_results = pd.merge(df_results.reset_index(), baseline_acc, left_on=[
            'dataset', 'depth'], right_index=True).groupby(['approach', 'dataset', 'depth']).sum()

        df_results.columns = ['obs_count', 'mean_train_acc', 'std_train_acc',
                              'mean_solv_time', 'std_solve_time', 'mean_gap',
                              'std_gap', 'mean_cart']

        return df_results

    def n_solved(output_df):
        output_df.status.fillna(np.inf, inplace=True)
        output_df.loc['status', :].groupby(
            ['approach', 'dataset', 'depth'])['status'].agg(['value_counts'])

        # If want to sort at a specific status...
        # status_df = output_df[output_df.loc[:, 'status'] == 2, :].groupby(['approach', 'dataset', 'depth'])[
        #     'status'].agg(['value_counts'])

        status_df = output_df.groupby(['approach', 'dataset', 'depth'])[
            'status'].agg(['value_counts'])
        return status_df


analyze = AnalyzeTime('aexperiment_time.csv')
output_df = analyze.load_dataset()
df_results = analyze.create_results(output_df)
