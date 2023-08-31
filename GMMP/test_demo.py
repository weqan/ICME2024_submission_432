import math
import pickle
import time

import numpy as np

from lib import GaussianMixtureModel


def train_gmm_plus(train_embeddings, train_labels, n_clusters):
    gmm_plus = GaussianMixtureModel(n_components=n_clusters).fit(train_embeddings)
    gmm_idxs = gmm_plus.predict(train_embeddings)

    gmm_dict = dict()
    gmm_labels_dict = dict()
    for i in range(n_clusters):
        gmm_no = np.where(gmm_idxs == i)
        gmm_dict[i] = train_embeddings[gmm_no[0]]
        gmm_labels_dict[i] = np.array(train_labels)[gmm_no[0]]
        print(i, gmm_dict[i].shape, gmm_labels_dict[i].shape)

    gmm_plus_rst = {"gmm_plus": gmm_plus, "gmm_dict": gmm_dict, "gmm_labels_dict": gmm_labels_dict}
    with open("./gmm_plus_rst/gmm_plus" + str(n_clusters) + "_rst.pickle", "wb") as gpr:
        pickle.dump(gmm_plus_rst, gpr)
        gpr.close()


# gmm_plus
def test_gmm_plus(test_embeddings, test_labels, n_clusters):
    with open("./gmm_plus_rst/gmm_plus" + str(n_clusters) + "_rst.pickle", "rb") as gpr:
        gpr = pickle.load(gpr)
        gmm_plus = gpr["gmm_plus"]
        gmm_dict = gpr["gmm_dict"]
        gmm_labels_dict = gpr["gmm_labels_dict"]

    test_pre_proba, test_pre_lpr = gmm_plus.predict_proba(test_embeddings)

    gmm_means = gmm_plus.means_
    center_pre_proba, center_pre_lpr = gmm_plus.predict_proba(gmm_means)

    center_lpr_list = []
    for j in range(center_pre_proba.shape[0]):
        center_lpr_list.append(center_pre_lpr[j][j])

    right_count = 0
    for i in range(len(test_labels)):
        test_embedding = test_embeddings[i]
        test_label = test_labels[i]
        test_proba = test_pre_proba[i]
        test_lpr = test_pre_lpr[i]

        top1 = np.argsort(test_proba)[-1]
        top2 = np.argsort(test_proba)[-2]

        test_inf_ent_list = []
        for j in range(len(test_lpr)):
            test_inf_ent_list.append(test_lpr[j])

        inf_ent_list = []
        for n in range(len(center_lpr_list)):
            test_inf_ent = test_inf_ent_list[n]
            if test_inf_ent <= 0:
                test_inf_ent = 1e-10
            center_lpr = center_lpr_list[n]
            tmp_inf_ent = np.sqrt(2 * math.log(center_lpr / test_inf_ent, 10))
            inf_ent_list.append(tmp_inf_ent)

        top_inf_ent_list = []
        for inf_idx in range(len(inf_ent_list)):
            if inf_idx in [top1, top2]:
                top_inf_ent_list.append(inf_ent_list[inf_idx])
            else:
                top_inf_ent_list.append(1)

        pre_idx = np.argmin(top_inf_ent_list)

        gmm_tmp_embeddings = gmm_dict[pre_idx]
        gmm_tmp_labels = gmm_labels_dict[pre_idx]
        gmm_var = test_embedding.dot(gmm_tmp_embeddings.T)
        gmm_closest_embedding_index = np.argmax(gmm_var)
        gmm_pre_label = gmm_tmp_labels[gmm_closest_embedding_index]

        if gmm_pre_label == test_label:
            right_count += 1

    print("acc:", right_count / len(test_labels))


if __name__ == "__main__":
    k = 3
    with open("../dataset/test.pickle", 'rb') as trf:
        train_file = pickle.load(trf)
        train_embeddings = train_file['embedding']
        train_labels = train_file['label']
    print("train_embeddings:", train_embeddings.shape)

    train_gmm_plus(train_embeddings, train_labels, k)

    with open("../dataset/train.pickle", 'rb') as tf:
        test_file = pickle.load(tf)
        test_embeddings = test_file['embedding']
        test_labels = test_file['label']
    print("test_embeddings:", test_embeddings.shape)

    test_gmm_plus(test_embeddings, test_labels, k)
