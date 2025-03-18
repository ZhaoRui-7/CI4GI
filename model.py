import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.fc1 = nn.Linear(input_dim, input_dim)
    
    def encode(self, ui_emb,ui_fangcha_emb, ui_hg, num_user, num_item):
        ui_emb = torch.mean(torch.stack([ui_emb], dim=0), dim=0)
        user_emb_i_side, _ = torch.split(ui_emb, [num_user, num_item], dim=0)
        mean = user_emb_i_side
        logvar = self.fc1(mean)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h = F.relu(z)
        return torch.sigmoid(h)
    
    def forward(self, ui_emb,ui_fangcha_emb, ui_hg, num_user, num_item):
        mean, logvar = self.encode(ui_emb, ui_fangcha_emb, ui_hg, num_user, num_item)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, logvar


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.normal_(self.a.data,std=0.1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # print(self.a[0])
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
        # return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        nout = nfeat//nheads
        self.attentions = [GraphAttentionLayer(nfeat, nfeat, dropout=dropout, alpha=alpha, concat=False) for _ in range(1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nfeat * 1, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        all_embeds = [x]
        x = torch.stack([att(x, adj) for att in self.attentions], dim=0)
        x = x.squeeze(0)
        return torch.mean(torch.stack(all_embeds, dim=0), dim=0)



class CollaborativeGCNConv(nn.Module):
    def __init__(self, n_layer, n_nodes):
        super(CollaborativeGCNConv, self).__init__()
        self.n_layer = n_layer
        self.n_nodes = n_nodes

    def forward(self, embed, edge_index, trend, return_final=False):
        agg_embed = embed
        all_embeds = [embed]

        row, col = edge_index

        for _ in range(self.n_layer):
            out = agg_embed[row] * trend.unsqueeze(-1)
            agg_embed = scatter(out, col, dim=0, dim_size=self.n_nodes, reduce='add')
            all_embeds.append(agg_embed)

        if return_final:
            return agg_embed

        return torch.mean(torch.stack(all_embeds, dim=0), dim=0)


class CI4GI(nn.Module):
    def __init__(self, num_user, num_item, num_group, emb_dim, layer, hg_dict, user_side_ssl, group_side_ssl,user_ssl_thershold,user_add_gitem_weight):
        super(CI4GI, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_group = num_group

        self.emb_dim = emb_dim
        self.layer = layer
        self.user_ssl_thershold = user_ssl_thershold
        self.user_add_gitem_weight = user_add_gitem_weight

        # Use when model training and testing on validation set
        self.user_hg_train = hg_dict["UserHGTrain"]
        self.user_hg_ssl = hg_dict["UserHGSSL"]
        # Use when model testing on test set
        self.user_hg_val = hg_dict["UserHGVal"]
        self.item_hg = hg_dict["ItemHG"]
        self.gi_hg_ssl = hg_dict["GIadj"]
        self.ui_trend, self.ui_edge,self.adj = hg_dict["UITrend"], hg_dict["UIEdge"],hg_dict["UIadj"]
        self.ui_add_hg = hg_dict["UIHGADD"]

        self.user_side_ssl = user_side_ssl
        self.group_side_ssl = group_side_ssl

        self.user_embedding_distinct = nn.Embedding(num_user, emb_dim)
        self.user_embedding_interest = nn.Embedding(num_user, emb_dim)
        self.user_embedding_interest_fangcha = nn.Embedding(num_user, emb_dim)
        self.group_embedding_distinct = nn.Embedding(num_group, emb_dim)
        self.group_embedding_interest = nn.Embedding(num_group, emb_dim)
        self.item_embedding = nn.Embedding(num_item, emb_dim)

        self.act = nn.Sigmoid()
        self.cgcn = CollaborativeGCNConv(1, num_user + num_item)
        self.gat = GAT(emb_dim,emb_dim,0.9,0.2,2)
        self.vae = VAE(emb_dim,emb_dim//2)

        self.u1_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.g1_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.u2_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.g2_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim))
        self.mlp_g = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim)
        )

        nn.init.normal_(self.user_embedding_distinct.weight, std=0.1)
        nn.init.normal_(self.user_embedding_interest.weight, std=0.1)
        nn.init.normal_(self.user_embedding_interest_fangcha.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.group_embedding_distinct.weight, std=0.1)
        nn.init.normal_(self.group_embedding_interest.weight, std=0.1)


    def compute(self, test_on_testset=False, return_other=False):
        """Forward Propagation"""
        user_hg = self.user_hg_train
        if test_on_testset:
            user_hg = self.user_hg_val
        
        ig_emb = torch.cat([self.item_embedding.weight, self.group_embedding_interest.weight], dim=0)
        
        for _ in range(self.layer):
            ig_emb = torch.sparse.mm(self.item_hg, ig_emb)
        tmp_item_embedding, group_emb_i_side = torch.split(ig_emb, [self.num_item, self.num_group], dim=0)

        ug_emb = torch.cat([self.user_embedding_distinct.weight, self.group_embedding_distinct.weight], dim=0)
        for _ in range(self.layer):
            ug_emb = torch.sparse.mm(user_hg, ug_emb)
        user_emb, group_emb_u_side = torch.split(ug_emb, [self.num_user, self.num_group], dim=0)
        # print(self.user_embedding_interest.weight[0][0])
        ui_emb = torch.cat([self.user_embedding_interest.weight, tmp_item_embedding], dim=0)
        ui_emb_final = self.gat(ui_emb,self.adj)
        user_emb_i_side, _ = torch.split(ui_emb_final, [self.num_user, self.num_item], dim=0)
        
        user_item = torch.matmul(self.user_hg_ssl,self.gi_hg_ssl) #U*G G*V
        user_item_embeds = torch.matmul(user_item.float(),tmp_item_embedding) #U*v V*d = U*d
        user_item_counts = user_item.sum(dim=1, keepdim=True).float()
        user_item_counts = user_item_counts.to_dense()
        user_item_counts = torch.where(user_item_counts == 0, torch.ones_like(user_item_counts), user_item_counts)
        user_item_avg_embeds = user_item_embeds / user_item_counts  # (U, d)

        user_emb_i_side = user_emb_i_side + self.user_add_gitem_weight* user_item_avg_embeds
        
        #分布表示
        ui_fangcha_emb = torch.cat([self.user_embedding_interest_fangcha.weight, tmp_item_embedding], dim=0)
        vae_reconstruct_u_emb,mean,logvar = self.vae(ui_emb,ui_emb,self.adj,self.num_user, self.num_item)

        x = torch.randn(self.num_user,1).to("cuda:0")  # 和 logvar 相同形状的随机数
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        
        final_u_emb = torch.cat([user_emb, user_emb_i_side], dim=1)
        final_g_emb = torch.cat([group_emb_u_side, group_emb_i_side], dim=1)

        
        if return_other:
            return final_u_emb, final_g_emb, user_emb_i_side, user_emb, group_emb_i_side, group_emb_u_side, self.gi_hg_ssl, self.user_hg_ssl,mean,logvar

        return final_u_emb, final_g_emb

    def forward(self, user_inputs, pos_groups, neg_groups):
        all_users, all_groups, all_user_emb_i_side, all_user_emb_g_side, all_group_emb_i_side, all_group_emb_u_side, gi_hg_ssl, user_hg,mean,logvar = self.compute(
            return_other=True)
        

        user_embeds = all_users[user_inputs]
        pos_embed = all_groups[pos_groups]
        neg_embed = all_groups[neg_groups]

        user_embeds_ego1, user_embeds_ego2 = self.user_embedding_interest(user_inputs), self.user_embedding_distinct(
            user_inputs)
        pos_embed_ego1, pos_emb_ego2 = self.group_embedding_interest(pos_groups), self.group_embedding_distinct(
            pos_groups)
        neg_embed_ego1, neg_emb_ego2 = self.group_embedding_interest(neg_groups), self.group_embedding_distinct(
            neg_groups)

        reg_loss = (1 / 2) * (
                user_embeds_ego1.norm(2).pow(2) + user_embeds_ego2.norm(2).pow(2) + pos_embed_ego1.norm(2).pow(
            2) + pos_emb_ego2.norm(2).pow(2) + neg_embed_ego1.norm(2).pow(2) + neg_emb_ego2.norm(2).pow(2)) \
                   / float(len(user_inputs))

        ssl_loss = 1
        if self.user_side_ssl:
            user_ssl_loss_1 = self.ssl_loss(all_user_emb_i_side[user_inputs], all_user_emb_g_side[user_inputs],
                                     all_user_emb_g_side,mean,logvar,user_inputs) + self.ssl_loss(all_user_emb_g_side[user_inputs],
                                                                          all_user_emb_i_side[user_inputs],
                                                                          all_user_emb_i_side,mean,logvar,user_inputs)
            user_ssl_loss_2 = self.ssl_loss(all_user_emb_i_side[user_inputs], all_user_emb_g_side[user_inputs],
                                     all_user_emb_g_side,None,None,user_inputs) + self.ssl_loss(all_user_emb_g_side[user_inputs],
                                                                          all_user_emb_i_side[user_inputs],
                                                                          all_user_emb_i_side,None,None,user_inputs)
        
        if self.group_side_ssl:
            group_ssl_loss_1 = 0
            group_ssl_loss_2 = self.ssl_loss(all_group_emb_i_side[pos_groups], all_group_emb_u_side[pos_groups],
                                       all_group_emb_u_side,None,None,pos_groups) + self.ssl_loss(all_group_emb_u_side[pos_groups],
                                                                             all_group_emb_i_side[pos_groups],
                                                                             all_group_emb_i_side,None,None,pos_groups)
            if ssl_loss is not None:
                ssl_loss_1 = user_ssl_loss_1 + group_ssl_loss_1
                ssl_loss_2 = user_ssl_loss_2 + group_ssl_loss_2
            else:
                ssl_loss_1 = group_ssl_loss_1
                ssl_loss_2 = group_ssl_loss_2
        return user_embeds, pos_embed, neg_embed, reg_loss, ssl_loss_1,ssl_loss_2,all_user_emb_i_side, all_user_emb_g_side#,all_users, all_groups

    
    def bpr_loss(self, user_input, pos_group_input, neg_group_input):
        """Loss computation using Cross-Entropy Loss"""
        (user_emb, pos_emb, neg_emb, reg_loss, twin_loss_1,twin_loss_2, all_user_emb_i_side, all_user_emb_g_side) = self.forward(user_input, pos_group_input, neg_group_input)
        
        # 计算正负样本的评分
        pos_score = torch.sum(user_emb * pos_emb, dim=-1)
        neg_score = torch.sum(user_emb * neg_emb, dim=-1)
        labels = torch.cat([torch.ones_like(pos_score), torch.full_like(neg_score,0)], dim=0)
        scores = torch.cat([pos_score, neg_score], dim=0)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))
        loss = bpr_loss

        return bpr_loss, reg_loss, twin_loss_1,twin_loss_2, all_user_emb_i_side, all_user_emb_g_side


    def get_user_rating(self, mode="val"):
        if mode == "val":
            all_users, all_groups = self.compute()
        elif mode == "test":
            all_users, all_groups = self.compute(test_on_testset=True)
        
        rating = self.act(torch.mm(all_users, all_groups.t()))
        return rating

    def compute_all_wasserstein_distances(self, mean, logvar):
        # 计算标准差
        std = torch.exp(0.5 * logvar)

        # 计算均值的差异（通过广播机制）
        mean_diff = mean.unsqueeze(1) - mean.unsqueeze(0)  # (num_users, num_users, embedding_size)
        mean_distance = torch.norm(mean_diff, p=2, dim=2)  # (num_users, num_users), 计算L2范数
        
        # 计算标准差的差异
        std_diff = std.unsqueeze(1) - std.unsqueeze(0)  # (num_users, num_users, embedding_size)
        cov_distance = torch.norm(std_diff, p=2, dim=2)  # (num_users, num_users), 计算L2范数
        
        # 计算 Wasserstein 距离
        wasserstein_distance = mean_distance + cov_distance  # (num_users, num_users)
        
        return wasserstein_distance
        
    def ssl_loss(self, user_emb_side1, user_emb_side2, user_emb_side2_all,mean,logvar,user_inputs):
        norm_user_side1 = F.normalize(user_emb_side1)
        norm_user_side2 = F.normalize(user_emb_side2)
        norm_all_user_side2 = F.normalize(user_emb_side2_all)

        pos_score = torch.mul(norm_user_side1, norm_user_side2).sum(dim=1)
        
        total_score = torch.matmul(norm_user_side1, norm_all_user_side2.transpose(0, 1))
        if mean!=None:
            #计算推土机距离
            wasserstein_distances = self.compute_all_wasserstein_distances(mean,logvar)

            mask = wasserstein_distances > self.user_ssl_thershold  
            mask.fill_diagonal_(True)
            batch_mask = mask[user_inputs, :] 
            masked_total_score = total_score.masked_fill(batch_mask == False, float('-inf'))
            total_score = masked_total_score

        pos_score = torch.exp(pos_score / 0.1)
        total_score = torch.exp(total_score / 0.1).sum(dim=1)
        ssl_loss = -torch.log(pos_score / total_score).sum()

        return ssl_loss / user_emb_side1.shape[0]


    