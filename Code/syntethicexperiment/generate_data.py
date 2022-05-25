import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
np.random.seed(69)


class GroundtruthDataset():
    def __init__(self) -> None:
        pass

    def generate_x_data(n_obs, n_features, n_categorical):
        data_list = []

        for feature in range(n_features):
            if feature < n_categorical:
                data_list.append(np.random.choice(
                    a=[0, 1, 2], size=n_obs, p=[0.5, 0.3, 0.2]))
            else:
                data_list.append(np.random.binomial(
                    1, np.random.uniform(0, 1), n_obs))
        return pd.DataFrame(data_list).T

    def add_target_variable(dataset):
        n_obs, _ = dataset.shape
        dataset['target'] = np.random.binomial(
            1, 0.5, n_obs)
        return dataset

    def encode_x_data(dataset, n_categorical):
        # TODO To prevent dependence remove one dummy variable?
        enc_dataset = pd.get_dummies(
            dataset, columns=range(int(n_categorical)))
        return enc_dataset

    def fit_tree(dataset, depth=4):
        clf = DecisionTreeClassifier(random_state=0, max_depth=depth,
                                     ccp_alpha=0)
        x_train = dataset.iloc[:, :-1]
        y_train = dataset.loc[:, 'target']
        clf.fit(x_train, y_train)
        return clf

    def retarget_dataset(clf, dataset):
        x_train = dataset.iloc[:, :-1]
        y_train = dataset.loc[:, 'target']

        x_in_leave = clf.apply(x_train)
        unique_leaves = np.unique(x_in_leave)
        skip_next = False
        for leaf_node in unique_leaves:
            if (leaf_node + 1) in unique_leaves:
                skip_next = True
                # Set pairs to left and right
                y_train.loc[np.where(x_in_leave == leaf_node)] = 0
                y_train.loc[np.where(x_in_leave == leaf_node + 1)] = 1
            elif skip_next:
                skip_next = False
                pass
            else:
                y_train.loc[np.where(x_in_leave == leaf_node)] = 0
        return y_train

    def generate_dataset(n_obs, n_features, percentage_binary):

        n_binary = np.floor(percentage_binary * n_features)
        n_categorical = n_features - n_binary

        dataset = GroundtruthDataset.generate_x_data(
            n_obs, n_features, n_categorical)
        dataset = GroundtruthDataset.encode_x_data(dataset, n_categorical)
        dataset = GroundtruthDataset.add_target_variable(dataset)

        return dataset

    def groundtruth_features(dataset, clf):
        features_clf = clf.tree_.feature
        features_clf = features_clf[features_clf >= 0]
        groundtruth_features = dataset.columns[features_clf]
        return groundtruth_features

    def groundtruth_dataset_and_features(dataset, depth):
        clf = GroundtruthDataset.fit_tree(dataset, depth)
        dataset.target = GroundtruthDataset.retarget_dataset(clf, dataset)
        grnd_features = GroundtruthDataset.groundtruth_features(dataset, clf)
        return dataset, grnd_features

    def get_dataset_and_features(n_obs, n_features, percentage_binary, depth):
        dataset = GroundtruthDataset.generate_dataset(
            n_obs, n_features, percentage_binary)
        retargeted_dataset, grnd_features = GroundtruthDataset.groundtruth_dataset_and_features(
            dataset, depth)

        grnd_features = [str(feature) for feature in grnd_features]
        return retargeted_dataset, grnd_features


GroundtruthDataset.get_dataset_and_features(100, 10, 0.3, 4)
