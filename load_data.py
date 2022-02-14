from __future__ import print_function, absolute_import, division

import numpy as np

from utils import toAbsolute, load_pickle, fix_list, split_train_test_clusters


def load_data(measure, setting, clu_thre, n_fold, max_sequence_length=512):
    # load data
    with open(toAbsolute('preprocessing/pdbbind_all_combined_input_'+measure),'rb') as f:
        data_pack = load_pickle(f)

    data_pack[7] = fix_list(data_pack[7])
    data_pack[8] = fix_list(data_pack[8])

    deleted_proteins = set()
    if max_sequence_length > 0:
        new_data_pack = [[] for x in range(len(data_pack))]
        max_vertex, max_sequence = 0, 0
        for index in range(len(data_pack[0])):
            max_vertex = max(max_vertex, len(data_pack[0][index]))
            max_sequence = max(max_sequence, len(data_pack[5][index]))
            if len(data_pack[5][index]) > max_sequence_length:
                deleted_proteins.add(data_pack[8][index])
                continue
            for i in range(len(new_data_pack)):
                new_data_pack[i].append(data_pack[i][index])


        data_pack = [ np.asarray(arr) for arr in new_data_pack]

    cid_list = data_pack[7]
    pid_list = data_pack[8]
    n_sample = len(cid_list)
    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    if setting == 'pit':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
            p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
            p_train = set(p_train_valid)-set(p_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if P_cluster_dict[pid_list[ele]] in p_train:
                    train_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_valid:
                    valid_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ', len(train_idx), 'test ', len(test_idx), 'valid ', len(valid_idx))

    elif setting == 'pid':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
        for fold in range(n_fold):
            c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
            c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
            c_train = set(c_train_valid)-set(c_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if C_cluster_dict[cid_list[ele]] in c_train:
                    train_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_valid:
                    valid_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ', len(train_idx), 'test ', len(test_idx), 'valid ', len(valid_idx))

    elif setting == 'ppi':
        assert n_fold ** 0.5 == int(n_fold ** 0.5)
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))

        for fold_x in range(int(n_fold ** 0.5)):
            for fold_y in range(int(n_fold ** 0.5)):
                c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
                c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
                c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid)/3), replace=False)
                c_train = set(c_train_valid)-set(c_valid)
                p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid)/3), replace=False)
                p_train = set(p_train_valid)-set(p_valid)

                train_idx, valid_idx, test_idx = [], [], []
                for ele in range(n_sample):
                    if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
                        train_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
                        valid_idx.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
                        test_idx.append(ele)
                train_idx_list.append(train_idx)
                valid_idx_list.append(valid_idx)
                test_idx_list.append(test_idx)
                print('fold', fold_x * int(n_fold ** 0.5) + fold_y, 'train ', len(train_idx), 'test ', len(test_idx),
                      'valid ', len(valid_idx))
    return data_pack, train_idx_list, valid_idx_list, test_idx_list