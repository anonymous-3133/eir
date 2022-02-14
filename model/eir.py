import math
import time
import torch
from torch import nn, einsum
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from utils import *

from .eir_graph import EIR_Graph
from .eir_attention import Attention






#define the model
class EIR(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, init_word_features, params, paramsExt):
        super(EIR, self).__init__()

        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features
        self.init_word_features = init_word_features
        """hyper part"""
        GNN_depth, inner_CNN_depth, attention_depth, DMA_depth, k_head, transformer_head, kernel_size, hidden_size1, hidden_size2, attention_hidden = params
        self.GNN_depth = GNN_depth
        self.inner_CNN_depth = inner_CNN_depth
        self.attention_depth = attention_depth
        self.DMA_depth = DMA_depth
        self.k_head = k_head
        self.transformer_head = transformer_head
        self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.attention_hidden = attention_hidden

        self.paramsExt = paramsExt

        self.graph = EIR_Graph(init_atom_features, init_bond_features, init_word_features, params, paramsExt)
        """CNN-RNN Module"""
        #CNN parameters
        self.embed_seq = nn.Embedding(len(self.init_word_features), 20, padding_idx=0)
        self.embed_seq.weight = nn.Parameter(self.init_word_features)
        self.embed_seq.weight.requires_grad = False

        self.embed_drop = torch.nn.Dropout(0.10)

        if self.inner_CNN_depth >= 0:
            #self.cnn_dropout = torch.nn.Dropout(0.10)
            self.conv_first = nn.Conv1d(20, self.hidden_size1, kernel_size=self.kernel_size,
                                            padding=(self.kernel_size - 1) // 2)

            self.conv_last = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size, padding=(self.kernel_size-1) //2)

            self.plain_CNN = nn.ModuleList([])
            for i in range(self.inner_CNN_depth):
                self.plain_CNN.append(nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size,  dilation=1, padding='same'))

        self.attention_dropout = self.paramsExt.attention_dropout
        if self.attention_depth > 0:

            self.trans_hid = nn.Linear(20, self.attention_hidden)

            self.transformer_encode = Attention(
                num_tokens = self.hidden_size1,
                dim = self.attention_hidden,
                depth = self.attention_depth,
                max_seq_len = self.paramsExt.max_sequence_length,
                absolute_pos_emb = False,
            )


        """Affinity Prediction Module"""
        self.super_final = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.c_final = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.p_final = nn.Linear(self.hidden_size1, self.hidden_size2)

        #DMA parameters
        self.mc0 = nn.Linear(hidden_size2, hidden_size2)
        self.mp0 = nn.Linear(hidden_size2, hidden_size2)

        self.mc1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
        self.mp1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])

        self.hc0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
        self.hp0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
        self.hc1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])

        self.c_to_p_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
        self.p_to_c_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])

        self.GRU_dma = nn.GRUCell(self.hidden_size2, self.hidden_size2)
        #Output layer
        self.W_out = nn.Linear(self.hidden_size2*self.hidden_size2*2, 1)

        """Pairwise Interaction Prediction Module"""
        self.pairwise_compound = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.pairwise_protein = nn.Linear(self.hidden_size1, self.hidden_size1)


    def mask_softmax(self,a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax


    def transformer_block(self, seq_mask, seq):
        src = F.gelu(self.trans_hid(seq))

        params = [src]
        params += [seq_mask > 0]

        output = self.transformer_encode(*params)
        return output

    def cnn_block(self, seq_mask, seq):
        seq = seq.transpose(1, 2)
        x = F.leaky_relu(self.conv_first(seq), 0.1)

        for i in range(self.inner_CNN_depth):
            x = self.plain_CNN[i](x)
            x = F.leaky_relu(x, 0.1)

        x = F.leaky_relu(self.conv_last(x), 0.1)

        H = x.transpose(1, 2)
        return H


    def CNN_module(self, batch_size, seq_mask, sequence):
        ebd = self.embed_seq(sequence)

        blocks = {
            'transformers': [self.transformer_block, self.attention_depth, None],
            'cnn': [self.cnn_block, self.inner_CNN_depth, None],
        }

        block_keys = ['transformers', 'cnn']

        for block_name in block_keys:
            block_func, depth, _ = blocks[block_name]
            if depth > 0:
                block_out = block_func(seq_mask, ebd)
                blocks[block_name][2] = block_out

        cnn_out = blocks['cnn'][2]
        transformer_out = blocks['transformers'][2]

        if self.attention_depth > 0 and self.inner_CNN_depth > 0:
            output = cnn_out + transformer_out
        elif self.attention_depth > 0:
            output = transformer_out
        else:
            output = cnn_out

        return output


    def Pairwise_pred_module(self, batch_size, comp_feature, prot_feature, vertex_mask, seq_mask):
        pairwise_c_feature = F.leaky_relu(self.pairwise_compound(comp_feature), 0.1)
        pairwise_p_feature = F.leaky_relu(self.pairwise_protein(prot_feature), 0.1)
        pairwise_pred = torch.sigmoid(torch.matmul(pairwise_c_feature, pairwise_p_feature.transpose(1,2)))
        pairwise_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))
        pairwise_pred = pairwise_pred*pairwise_mask

        return pairwise_pred


    def Affinity_pred_module(self, batch_size, comp_feature, prot_feature, super_feature, vertex_mask, seq_mask, pairwise_pred):
        comp_feature = F.leaky_relu(self.c_final(comp_feature), 0.1)
        prot_feature = F.leaky_relu(self.p_final(prot_feature), 0.1)
        super_feature = F.leaky_relu(self.super_final(super_feature.view(batch_size,-1)), 0.1)

        cf, pf = self.dma_gru(batch_size, comp_feature, vertex_mask, prot_feature, seq_mask, pairwise_pred)

        cf = torch.cat([cf.view(batch_size,-1), super_feature.view(batch_size,-1)], dim=1)
        kroneck = F.leaky_relu(torch.matmul(cf.view(batch_size,-1,1), pf.view(batch_size,1,-1)).view(batch_size,-1), 0.1)

        affinity_pred = self.W_out(kroneck)
        return affinity_pred


    def dma_gru(self, batch_size, comp_feats, vertex_mask, prot_feats, seq_mask, pairwise_pred):
        vertex_mask = vertex_mask.view(batch_size,-1,1)
        seq_mask = seq_mask.view(batch_size,-1,1)

        c0 = torch.sum(comp_feats*vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        p0 = torch.sum(prot_feats*seq_mask, dim=1) / torch.sum(seq_mask, dim=1)

        m = c0*p0
        for DMA_iter in range(self.DMA_depth):
            c_to_p = torch.matmul(pairwise_pred.transpose(1,2), torch.tanh(self.c_to_p_transform[DMA_iter](comp_feats)))  # batch * n_residue * hidden
            p_to_c = torch.matmul(pairwise_pred, torch.tanh(self.p_to_c_transform[DMA_iter](prot_feats)))  # batch * n_vertex * hidden

            c_tmp = torch.tanh(self.hc0[DMA_iter](comp_feats))*torch.tanh(self.mc1[DMA_iter](m)).view(batch_size,1,-1)*p_to_c
            p_tmp = torch.tanh(self.hp0[DMA_iter](prot_feats))*torch.tanh(self.mp1[DMA_iter](m)).view(batch_size,1,-1)*c_to_p

            c_att = self.mask_softmax(self.hc1[DMA_iter](c_tmp).view(batch_size,-1), vertex_mask.view(batch_size,-1))
            p_att = self.mask_softmax(self.hp1[DMA_iter](p_tmp).view(batch_size,-1), seq_mask.view(batch_size,-1))

            cf = torch.sum(comp_feats*c_att.view(batch_size,-1,1), dim=1)
            pf = torch.sum(prot_feats*p_att.view(batch_size,-1,1), dim=1)

            m = self.GRU_dma(m, cf*pf)

        return cf, pf


    def forward(self, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence):
        batch_size = vertex.size(0)

        atom_feature, super_feature = self.graph(batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask)
        prot_feature = self.CNN_module(batch_size, seq_mask, sequence)

        pairwise_pred = self.Pairwise_pred_module(batch_size, atom_feature, prot_feature, vertex_mask, seq_mask)
        affinity_pred = self.Affinity_pred_module(batch_size, atom_feature, prot_feature, super_feature, vertex_mask, seq_mask, pairwise_pred)

        return affinity_pred, pairwise_pred

