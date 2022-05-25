from utilities.utils import Utils
from utilities.utils_bin_oct import UtilsBinOCT
from utilities.utils_oct import UtilsOCT


def features_used(grb_model):
    tree = grb_model.tree

    try:
        b_value = grb_model.model.getAttr("X", grb_model.b)
        beta_value = grb_model.model.getAttr("X", grb_model.beta)
        p_value = grb_model.model.getAttr("X", grb_model.p)
    except:
        return []

    selected_feature_list = []
    # TODO THIS IS Coded very stupidly this is a static class...
    utils = Utils()
    for n in tree.Nodes + tree.Leaves:
        _, _, selected_feature, _, _ = utils.get_node_status(
            grb_model, b_value, beta_value, p_value, n)

        if selected_feature is not None:
            selected_feature_list.append(selected_feature)
    return selected_feature_list


def features_used_bin(grb_model):
    tree = grb_model.tree
    try:
        b_value = grb_model.model.getAttr("X", grb_model.f)
        beta_value = grb_model.model.getAttr("X", grb_model.p)
        p_value = 1  # We always split in binary tree
    except:
        return []

    selected_feature_list = []
    # TODO THIS IS Coded very stupidly this is a static class...
    utils = UtilsBinOCT()
    for n in tree.Nodes + tree.Leaves:
        _, _, selected_feature, _, _ = utils.get_node_status(
            grb_model, b_value, beta_value, p_value, n)

        if selected_feature is not None:
            selected_feature_list.append(selected_feature)
    return selected_feature_list


def features_used_oct(grb_model):
    tree = grb_model.tree

    try:
        b_value = grb_model.model.getAttr("X", grb_model.b)
        beta_value = grb_model.model.getAttr("X", grb_model.beta)
        p_value = grb_model.model.getAttr("X", grb_model.p)
    except:
        return []

    selected_feature_list = []
    # TODO THIS IS Coded very stupidly this is a static class...
    utils = UtilsOCT()
    for n in tree.Nodes + tree.Leaves:
        _, _, selected_feature, _, _ = utils.get_node_status(
            grb_model, b_value, beta_value, p_value, n)

        if selected_feature is not None:
            selected_feature_list.append(selected_feature)
    return selected_feature_list


def features_used_cart(clf):
    features_clf = clf.tree_.feature
    features_clf = features_clf[features_clf >= 0]
    return features_clf
