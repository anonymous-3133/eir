import math
import time
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from utils import *





from utils import elem_list
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


class EIR_Graph(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, init_word_features, params, paramsExt):
        super(EIR_Graph, self).__init__()
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

        """GraphConv Module"""
        self.vertex_embedding = nn.Linear(atom_fdim, self.hidden_size1) #first transform vertex features into hidden representations

        # GWM parameters
        self.W_a_main = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_a_super = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_main = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_bmm = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, 1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])

        self.W_super = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_main_to_super = nn.ModuleList([nn.Linear(self.hidden_size1*self.k_head, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_super_to_main = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])

        self.W_zm1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zm2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zs1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zs2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.GRU_main = nn.GRUCell(self.hidden_size1, self.hidden_size1)
        self.GRU_super = nn.GRUCell(self.hidden_size1, self.hidden_size1)

        # WLN parameters
        self.label_U2 = nn.ModuleList([nn.Linear(self.hidden_size1+bond_fdim, self.hidden_size1) for i in range(self.GNN_depth)]) #assume no edge feature transformation
        self.label_U1 = nn.ModuleList([nn.Linear(self.hidden_size1*2, self.hidden_size1) for i in range(self.GNN_depth)])



    def mask_softmax(self,a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax



    def wln_unit(self, batch_size, vertex_mask, vertex_features, edge_initial, atom_adj, bond_adj, nbs_mask, GNN_iter):
        n_vertex = vertex_mask.size(1)
        n_nbs = nbs_mask.size(2)

        vertex_mask = vertex_mask.view(batch_size,n_vertex,1)
        nbs_mask = nbs_mask.view(batch_size,n_vertex,n_nbs,1)

        vertex_nei = torch.index_select(vertex_features.view(-1, self.hidden_size1), 0, atom_adj).view(batch_size, n_vertex, n_nbs,self.hidden_size1)
        edge_nei = torch.index_select(edge_initial.view(-1, bond_fdim), 0, bond_adj).view(batch_size,n_vertex,n_nbs,bond_fdim)

        l_nei = torch.cat((vertex_nei, edge_nei), -1)
        nei_label = F.leaky_relu(self.label_U2[GNN_iter](l_nei), 0.1)
        nei_label = torch.sum(nei_label*nbs_mask, dim=-2)
        new_label = torch.cat((vertex_features, nei_label), 2)
        new_label = self.label_U1[GNN_iter](new_label)
        vertex_features = F.leaky_relu(new_label, 0.1)

        return vertex_features



    def forward(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
        n_vertex = vertex_mask.size(1)

        # initial features
        vertex_initial = torch.index_select(self.init_atom_features, 0, vertex.view(-1))
        vertex_initial = vertex_initial.view(batch_size, -1, atom_fdim)
        edge_initial = torch.index_select(self.init_bond_features, 0, edge.view(-1))
        edge_initial = edge_initial.view(batch_size, -1, bond_fdim)

        vertex_feature = F.leaky_relu(self.vertex_embedding(vertex_initial), 0.1)
        super_feature = torch.sum(vertex_feature*vertex_mask.view(batch_size,-1,1), dim=1, keepdim=True)

        for GWM_iter in range(self.GNN_depth):
            # prepare main node features
            for k in range(self.k_head):
                a_main = torch.tanh(self.W_a_main[GWM_iter][k](vertex_feature))
                a = self.W_bmm[GWM_iter][k](a_main*super_feature)
                attn = self.mask_softmax(a.view(batch_size,-1), vertex_mask).view(batch_size,-1,1)
                k_main_to_super = torch.bmm(attn.transpose(1,2), self.W_main[GWM_iter][k](vertex_feature))
                if k == 0:
                    m_main_to_super = k_main_to_super
                else:
                    m_main_to_super = torch.cat([m_main_to_super, k_main_to_super], dim=-1)  # concat k-head
            main_to_super = torch.tanh(self.W_main_to_super[GWM_iter](m_main_to_super))
            main_self = self.wln_unit(batch_size, vertex_mask, vertex_feature, edge_initial, atom_adj, bond_adj, nbs_mask, GWM_iter)

            super_to_main = torch.tanh(self.W_super_to_main[GWM_iter](super_feature))
            super_self = torch.tanh(self.W_super[GWM_iter](super_feature))
            # warp gate and GRU for update main node features, use main_self and super_to_main
            z_main = torch.sigmoid(self.W_zm1[GWM_iter](main_self) + self.W_zm2[GWM_iter](super_to_main))
            hidden_main = (1-z_main)*main_self + z_main*super_to_main
            vertex_feature = self.GRU_main(hidden_main.view(-1, self.hidden_size1), vertex_feature.view(-1, self.hidden_size1))
            vertex_feature = vertex_feature.view(batch_size, n_vertex, self.hidden_size1)
            # warp gate and GRU for update super node features
            z_supper = torch.sigmoid(self.W_zs1[GWM_iter](super_self) + self.W_zs2[GWM_iter](main_to_super))
            hidden_super = (1-z_supper)*super_self + z_supper*main_to_super
            super_feature = self.GRU_super(hidden_super.view(batch_size, self.hidden_size1), super_feature.view(batch_size, self.hidden_size1))
            super_feature = super_feature.view(batch_size, 1, self.hidden_size1)

        return vertex_feature, super_feature