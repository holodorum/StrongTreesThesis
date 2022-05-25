import numpy as np


def discovery_rate(ground_features, used_features):
    count_true_discovery = 0
    count_false_discovery = 0

    n_ground_features = len(ground_features)
    n_used_features = len(used_features)

    if n_used_features == 0:
        return np.NaN, np.NaN

    ground_features_copy = ground_features.copy()
    for used_feature in used_features:
        if used_feature in ground_features_copy:
            count_true_discovery += 1
            ground_features_copy.pop(ground_features_copy.index(used_feature))
        else:
            count_false_discovery += 1

    true_discovery_rate = count_true_discovery/float(n_ground_features)
    false_discovery_rate = count_false_discovery/float(n_used_features)

    if true_discovery_rate == 1 and false_discovery_rate > 0:
        print('whatsup?')
    return true_discovery_rate, false_discovery_rate
