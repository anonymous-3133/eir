import argparse
import datetime
import json
import platform
from collections import OrderedDict

from tqdm import tqdm

from eir import *


def limitNumberSize(num):
    if num > 100000:
        limited_number = "{:.3e}".format(num)
    else:
        limited_number = str(num)
    return limited_number


def train_and_eval(train_data, valid_data, test_data, params, paramsExt, batch_size=32, num_epoch=30):
    init_A, init_B, init_W = loading_emb(measure)
    net = EIR(init_A, init_B, init_W, params, paramsExt)
    net.cuda()
    net.apply(weights_init)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss()

    net.parameters()
    params = net.parameters()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=0.0005, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    min_rmse = 1000

    for epoch in range(num_epoch):
        total_loss = 0
        affinity_loss = 0
        pairwise_loss = 0

        net.train()

        with tqdm(train_data, unit="batch") as tepoch:

            with tqdm(total=len(tepoch), bar_format="{postfix}") as line2:

                for i, (
                        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, affinity_label,
                        pairwise_mask,
                        pairwise_label) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    optimizer.zero_grad()
                    affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask,
                                                       seq_mask, sequence)

                    loss_aff = criterion1(affinity_pred, affinity_label)
                    loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)

                    loss = loss_aff + 0.1 * loss_pairwise

                    total_loss += float(loss.data * batch_size)
                    affinity_loss += float(loss_aff.data * batch_size)
                    pairwise_loss += float(loss_pairwise.data * batch_size)

                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5)
                    optimizer.step()

                    divider = float(i + 1)

                    postfix = OrderedDict({
                        'loss': limitNumberSize(round(total_loss / divider, 4)),
                        'affinity loss': limitNumberSize(round(affinity_loss / divider, 4)),
                        'pairwise loss': limitNumberSize(round(pairwise_loss / divider, 4)),
                        'lr': optimizer.param_groups[0]['lr']
                    })

                    line2.set_postfix(postfix,
                                      refresh=False
                                      )
                    line2.update()
        scheduler.step()
        net.eval()

        perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']

        valid_performance, valid_label, valid_output = test(net, valid_data)
        print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
        print('valid', len(valid_output), ' '.join(print_perf))

        if valid_performance[0] < min_rmse:
            min_rmse = valid_performance[0]
            test_performance, test_label, test_output = test(net, test_data)
        print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]

    print('Finished Training')
    print('test ', len(test_output), ' '.join(print_perf))
    return test_performance, test_label, test_output


@torch.no_grad()
def test(net, test_data):
    output_gpu = torch.FloatTensor([]).cuda()
    label_gpu = torch.FloatTensor([]).cuda()
    pairwise_auc_list = []

    for i, (vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, affinity_label, pairwise_mask,
            pairwise_label) in enumerate(test_data):

        affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

        for j in range(len(pairwise_mask)):
            if pairwise_mask[j]:
                num_vertex = int(torch.sum(vertex_mask[j, :]))
                num_residue = int(torch.sum(seq_mask[j, :]))
                pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
                pairwise_label_i = pairwise_label[j, :num_vertex, :num_residue].reshape(-1).cpu()
                pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
        output_gpu = torch.cat((output_gpu, affinity_pred), 0)
        label_gpu = torch.cat((label_gpu, affinity_label), 0)

    rmse_value_gpu, pearson_value_gpu, spearman_value_gpu = reg_scores_gpu(label_gpu, output_gpu)

    average_pairwise_auc = np.mean(pairwise_auc_list)

    test_performance = [rmse_value_gpu, pearson_value_gpu, spearman_value_gpu, average_pairwise_auc]
    return test_performance, label_gpu, output_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run arguments.')
    parser.add_argument('measure', choices=['IC50', 'KIKD'])
    parser.add_argument('setting', choices=['pid', 'pit', 'ppi'])
    parser.add_argument('clu_thre', choices=['0.3', '0.4', '0.5', '0.6'])
    parser.add_argument('--maxlen', dest='max_sequence_length', type=int, default=3072,
                        help='Maximum protein sequence length')
    parser.add_argument('--attention_depth', dest='attention_depth', type=int, default=2,
                        help='Attention depth')
    parser.add_argument('--attention_hidden', dest='attention_hidden', type=int, default=256)
    parser.add_argument('--attention_dropout', dest='attention_dropout', type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--name", default='run')

    args = parser.parse_args()
    measure = args.measure
    setting = args.setting
    clu_thre = float(args.clu_thre)

    n_epoch = 30
    n_rep = 5

    assert setting in ['pid', 'pit', 'ppi']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD']
    GNN_depth, inner_CNN_depth, attention_depth, DMA_depth = 4, 2, args.attention_depth, 2
    attention_hidden, hidden_size1, hidden_size2 =  args.attention_hidden, 128, 128
    n_fold = 5
    if setting == 'pid':
        k_head, transformer_head, kernel_size = 2, 4, 7
    elif setting == 'pit':
        k_head, transformer_head, kernel_size = 1, 2, 5
    elif setting == 'ppi':
        n_fold = 9
        k_head, transformer_head, kernel_size= 1, 2, 7
    else:
        raise NotImplemented(f'Setting {setting} not implemented')

    batch_size = args.batch_size

    params = [GNN_depth, inner_CNN_depth, attention_depth, DMA_depth, k_head, transformer_head, kernel_size,
              hidden_size1, hidden_size2, attention_hidden]

    print('Dataset: PDBbind v2018 with measurement', measure)
    print('Clustering threshold:', clu_thre)
    print('Number of epochs:', n_epoch)
    print('Number of repeats:', n_rep)

    num_workers = 0
    rep_all_list = []
    rep_avg_list = []
    for a_rep in range(n_rep):
        data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold,
                                                                             max_sequence_length=args.max_sequence_length)
        fold_score_list = []

        for a_fold in range(n_fold):
            print('repeat', a_rep + 1, 'fold', a_fold + 1, 'begin')
            train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
            print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))

            train_data = data_from_index(data_pack, train_idx)
            train_data = ProteinDataset(train_data)
            train_data = torch.utils.data.DataLoader(train_data,
                                                     collate_fn=batch_data_process,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=False,
                                                     shuffle=True)

            valid_data = data_from_index(data_pack, valid_idx)
            valid_data = ProteinDataset(valid_data)
            valid_data = torch.utils.data.DataLoader(valid_data,
                                                     collate_fn=batch_data_process,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=False,
                                                     shuffle=True)

            test_data = data_from_index(data_pack, test_idx)
            test_data = ProteinDataset(test_data)
            test_data = torch.utils.data.DataLoader(test_data,
                                                    collate_fn=batch_data_process,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=False,
                                                    shuffle=True)

            test_performance, test_label, test_output = train_and_eval(train_data, valid_data, test_data, params, args,
                                                                       batch_size, n_epoch)
            rep_all_list.append(test_performance)
            fold_score_list.append(test_performance)
            print('-' * 30)
        print('fold avg performance', np.mean(fold_score_list, axis=0))
        rep_avg_list.append(np.mean(fold_score_list, axis=0))

    print('all repetitions done')
    print('print all stats: RMSE, Pearson, Spearman, avg pairwise AUC')
    print('mean', np.mean(rep_all_list, axis=0))
    print('std', np.std(rep_all_list, axis=0))
    print('==============')
    print('print avg stats:  RMSE, Pearson, Spearman, avg pairwise AUC')
    print('mean', np.mean(rep_avg_list, axis=0))
    print('std', np.std(rep_avg_list, axis=0))
    print('Hyper-parameters:', [para_names[i] + ':' + str(params[i]) for i in range(7)])
