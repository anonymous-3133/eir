import math
import time
import torch
from torch import nn, einsum
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from utils import *

from utils import elem_list
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6



# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# blocks
def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = None,
        pos_emb = None
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # rotary positional embedding
        assert not (exists(pos_emb) and not exists(max_seq_len)), 'max_seq_len must be passed in if to use rotary positional embeddings'

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # if using relative positional encoding, make sure to reduce pairs of consecutive feature dimension before doing projection to attention logits

        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head // kv_attn_proj_divisor, 1, bias = False)  # for projecting keys to key attention logits

        # final transformation of values to "r" as in the paper
        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, device, h, use_rotary_emb = x.shape[1], x.device, self.heads, exists(self.pos_emb)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # if relative positional encoding is needed

        if use_rotary_emb:
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
            freqs = rearrange(freqs[:n], 'n d -> () () n d')
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token

        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token

        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        u = v_aggr * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # transformation step

        r = self.to_r(u)

        # paper then says to add the queries as a residual

        r = r + q

        # combine heads

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)

# main class
class Attention(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        absolute_pos_emb = False
    ):
        super().__init__()
        # positional embeddings

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, pos_emb = layer_pos_emb, max_seq_len = max_seq_len)
            ff = FeedForward(dim, mult = ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers

        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)

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


    def GraphConv_module(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
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

        atom_feature, super_feature = self.GraphConv_module(batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask)
        prot_feature = self.CNN_module(batch_size, seq_mask, sequence)

        pairwise_pred = self.Pairwise_pred_module(batch_size, atom_feature, prot_feature, vertex_mask, seq_mask)
        affinity_pred = self.Affinity_pred_module(batch_size, atom_feature, prot_feature, super_feature, vertex_mask, seq_mask, pairwise_pred)

        return affinity_pred, pairwise_pred

