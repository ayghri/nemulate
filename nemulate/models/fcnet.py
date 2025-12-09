from typing import List, Tuple, Dict
from sklearn.model_selection import KFold
from torch import nn
from sshcast.models.utils import build_network


def fcnet_no_cluster(layers_configs: List[Tuple[str, Dict]]) -> nn.Module:
    return build_network(layers_configs)


def cross_validation(
    inputs,
    targets,
    weight_map,
    clustering,
    dropout_rate,
    epochs,
    lr,
    folder_saving,
    montecarlo_dropout=False,
    cluster_index=None,
):

    # define the K-fold Cross Validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # loss_per_fold = []

    # input_shape = inputs.shape[1]
    fold_no = 1
    for train, test in kfold.split(inputs, targets):

        print("fold num: ", fold_no)

        ## get the weights for train and test
        weight_map_train = weight_map[train]
        weight_map_test = weight_map[test]

        print(
            len(train),
            len(weight_map_train),
            inputs[train].shape,
            len(test),
            len(weight_map_test),
            inputs[test].shape,
        )

        # define the model architecture (for the different partitioning schemes)
        # if clustering == "no_cluster":
        #     inp = Input(
        #         input_shape,
        #     )
        #     x = Dense(1024, activation="relu", kernel_regularizer=l2(0.000005))(inp)
        #     x = Dense(512, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #     x = get_dropout(x, p=dropout_rate, mc=montecarlo_dropout)
        #     x = Dense(256, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #     out = Dense(1)(x)
        #
        #     model = Model(inputs=inp, outputs=out)
        #
        # elif clustering == "hardcoded":
        #     inp = Input(
        #         input_shape,
        #     )
        #
        #     if cluster_index == 2:
        #         x = Dense(256, activation="relu", kernel_regularizer=l2(0.000005))(inp)
        #         x = Dense(64, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         x = get_dropout(x, p=dropout_rate, mc=montecarlo_dropout)
        #         out = Dense(1)(x)
        #
        #         model = Model(inputs=inp, outputs=out)
        #
        #     else:
        #         x = Dense(1024, activation="relu", kernel_regularizer=l2(0.000005))(inp)
        #         x = Dense(512, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         x = get_dropout(x, p=dropout_rate, mc=montecarlo_dropout)
        #         x = Dense(256, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         out = Dense(1)(x)
        #
        #         model = Model(inputs=inp, outputs=out)
        #
        # else:
        #     inp = Input(
        #         input_shape,
        #     )
        #     smaller_clusters_list = [
        #         2,
        #         3,
        #     ]  ## smaller networks are chosen for smaller clusters
        #     # since we experiment 2,4,8,16,32 and 64 number of clusters for the spectral case we modify this list accordingly
        #     # after taking a look at the cluster sizes generated.
        #     if cluster_index in smaller_clusters_list:
        #         x = Dense(256, activation="relu", kernel_regularizer=l2(0.000005))(inp)
        #         x = Dense(128, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         x = get_dropout(x, p=dropout_rate, mc=montecarlo_dropout)
        #         out = Dense(1)(x)
        #
        #         model = Model(inputs=inp, outputs=out)
        #
        #     else:
        #         x = Dense(1024, activation="relu", kernel_regularizer=l2(0.000005))(inp)
        #         x = Dense(512, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         x = get_dropout(x, p=dropout_rate, mc=montecarlo_dropout)
        #         x = Dense(256, activation="relu", kernel_regularizer=l2(0.000005))(x)
        #         out = Dense(1)(x)
        #
        #         model = Model(inputs=inp, outputs=out)
