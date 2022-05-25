from generate_data import GroundtruthDataset
from generate_data import *


def test_size_generate_x_data():
    n_obs = 100
    n_features = 30
    n_categorical = 25

    dataset = GroundtruthDataset.generate_x_data(
        n_obs, n_features, n_categorical)
    assert dataset.shape == (n_obs, n_features)
    assert type(dataset) == pd.DataFrame


def test_size_encoded_x_data():
    n_obs = 100
    n_features = 30
    n_categorical = 25

    dataset = GroundtruthDataset.generate_x_data(
        n_obs, n_features, n_categorical)
    enc_dataset = GroundtruthDataset.encode_x_data(dataset, n_categorical)
    assert enc_dataset.shape == (n_obs, (n_features + n_categorical*2))


def test_size_full_data():
    n_obs = 100
    n_features = 30
    n_categorical = 25

    dataset = GroundtruthDataset.generate_x_data(
        n_obs, n_features, n_categorical)
    enc_dataset = GroundtruthDataset.encode_x_data(dataset, n_categorical)
    full_dataset = GroundtruthDataset.add_target_variable(enc_dataset)
    assert full_dataset.shape == (n_obs, (n_features + n_categorical*2 + 1))


def test_similar_x_retargeted_dataset():
    n_obs, n_features, percentage_binary = (1000, 15, 0.8)
    dataset = GroundtruthDataset.generate_dataset(
        n_obs, n_features, percentage_binary)
    retargeted_dataset, features = GroundtruthDataset.groundtruth_dataset_and_features(
        dataset, depth=4)
    assert dataset.iloc[:, :-1].equals(retargeted_dataset.iloc[:, :-1])
    assert dataset.target.equals(retargeted_dataset.target)

# # TODO WHy NOT THE SAME RESULT IN SPITE OF RANDOM SEED?
# def test_repeatability_with_random_seed():
#     n_obs = 100
#     n_features = 30
#     percentage_binary = 0.5

#     dataset_1 = generate_dataset(n_obs, n_features, percentage_binary)
#     dataset_2 = generate_dataset(n_obs, n_features, percentage_binary)
#     assert dataset_1.equals(dataset_2)
