import pandas as pd
import numpy as np
from torch import nn as nn
import torch
import os
from torch_geometric.nn import GCNConv
from SKADSC import SKAttention, ResNetDWConv2d
from BAN import BANLayer, MLP_Projector
import torch.nn.functional as F


def pro_data(data, em, ed):
    """
    根据边信息从节点特征矩阵中提取相应的特征向量

    参数:
    data: 边信息张量，形状为[num_edges, 2]，每一行表示一条边的两个节点索引
    em: miRNA节点特征矩阵，形状为[num_miRNA, feature_dim]
    ed: circRNA节点特征矩阵，形状为[num_circRNA, feature_dim]

    返回:
    Em: 根据边信息提取的miRNA特征向量，形状为[num_edges, feature_dim]
    Ed: 根据边信息提取的circRNA特征向量，形状为[num_edges, feature_dim]
    """
    # 转置边数据，便于索引操作
    edgeData = data.t()

    # 保存原始特征数据
    mFeaData = em
    dFeaData = ed

    m_index = edgeData[0]   # 提取miRNA索引（边数据的第一行）
    d_index = edgeData[1]   # 提取circRNA索引（边数据的第二行）

    Em = torch.index_select(mFeaData, 0, m_index)   # 根据索引从miRNA特征矩阵中选择对应的特征
    Ed = torch.index_select(dFeaData, 0, d_index)   # 根据索引从circRNA特征矩阵中选择对应的特征

    return Em, Ed


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        # self.out = nn.Linear(inSize + inSize + inSize, outSize)
        self.out = nn.Linear(inSize * 2, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)  # batchsize*featuresize
        if self.outBn: x = self.bns(x) if len(x.shape) == 2 else self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


# model control


# model control
class MSCLCMI(nn.Module):
    def __init__(self, args):
        super(MSCLCMI, self).__init__()
        self.args = args
        self.mc_graph_dataset = dict()
        # miRNA feature

        self.miRNA_doc2vec = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\miRNA_doc2vec_dict.txt'),
                                         header=None).iloc[:, 1:].values
        self.miRNA_role2vec = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\miRNA_role2vec_dict.txt'),
                                          header=None).iloc[:, 1:].values
        self.miRNA_ernie = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\miRNA_RNAErnie_PCA.csv'),
                                       header=None).iloc[:, 1:].values
        self.miRNA_tpcp = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\miRNA_tpcp.txt'),
                                      header=None).iloc[:, 1:].values
        self.miRNA_squence = np.loadtxt(
            os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\miRNA_sequence_similarity.csv'), delimiter=',')

        # miRNA simi graph
        self.miRNA_simlarity_graph = torch.LongTensor(np.argwhere(self.miRNA_squence > self.args.r).T)
        self.mc_graph_dataset['mm_doc'] = {'data_matrix': self.miRNA_doc2vec, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_role'] = {'data_matrix': self.miRNA_role2vec, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_tpcp'] = {'data_matrix': self.miRNA_tpcp, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_ernie'] = {'data_matrix': self.miRNA_ernie, 'edges': self.miRNA_simlarity_graph}

        from torch_geometric.nn import GATConv

        self.gcn_x_m_doc = GATConv(self.args.doc_dim, self.args.doc_dim, heads=1, concat=False)
        self.gcn_x_m_role = GATConv(self.args.role_dim, self.args.role_dim, heads=1, concat=False)
        self.gcn_x_m_tpcp = GATConv(self.args.tpcp_dim, self.args.tpcp_dim, heads=1, concat=False)
        self.gcn_x_m_ernie = GATConv(self.args.ernie_dim, self.args.ernie_dim, heads=1, concat=False)


        self.gcn_x_m_doc_p = nn.Linear(self.args.doc_dim, self.args.pro_dim)
        self.gcn_x_m_role_p = nn.Linear(self.args.role_dim, self.args.pro_dim)
        self.gcn_x_m_tpcp_p = nn.Linear(self.args.tpcp_dim, self.args.pro_dim)
        self.gcn_x_m_ernie_p = nn.Linear(self.args.ernie_dim, self.args.pro_dim)

        self.cnn_m = nn.Conv2d(in_channels=self.args.channel_num, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

        # circRNA feature
        self.circRNA_doc2vec = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\circRNA_doc2vec_dict.txt'),
                                           header=None).iloc[:, 1:].values
        self.circRNA_role2vec = pd.read_csv(
            os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\circRNA_role2vec_dict.txt'), header=None).iloc[:,
                                1:].values
        self.circRNA_ernie = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\circRNA_RNAErnie_PCA.csv'),
                                         header=None).iloc[:, 1:].values
        self.circRNA_tpcp = pd.read_csv(os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\circRNA_tpcp.txt'),
                                        header=None).iloc[:, 1:].values
        self.circRNA_squence = np.loadtxt(
            os.path.join(r'D:\YD\MFERL-main\datasets\CircBANK\circRNA_sequence_similarity.csv'), delimiter=',')

        # circRNA simi graph
        self.circRNA_simlarity_graph = torch.LongTensor(np.argwhere(self.circRNA_squence > self.args.r).T)
        self.mc_graph_dataset['cc_doc'] = {'data_matrix': self.circRNA_doc2vec, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_role'] = {'data_matrix': self.circRNA_role2vec, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_tpcp'] = {'data_matrix': self.circRNA_tpcp, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_ernie'] = {'data_matrix': self.circRNA_ernie, 'edges': self.circRNA_simlarity_graph}


        self.gcn_x_c_doc = GATConv(self.args.doc_dim, self.args.doc_dim, heads=1, concat=False)
        self.gcn_x_c_role = GATConv(self.args.role_dim, self.args.role_dim, heads=1, concat=False)
        self.gcn_x_c_tpcp = GATConv(self.args.tpcp_dim, self.args.tpcp_dim, heads=1, concat=False)
        self.gcn_x_c_ernie = GATConv(self.args.ernie_dim, self.args.ernie_dim, heads=1, concat=False)

        self.gcn_x_c_kmer_p = nn.Linear(self.args.kmer_dim, self.args.pro_dim)
        self.gcn_x_c_doc_p = nn.Linear(self.args.doc_dim, self.args.pro_dim)
        self.gcn_x_c_role_p = nn.Linear(self.args.role_dim, self.args.pro_dim)
        self.gcn_x_c_tpcp_p = nn.Linear(self.args.tpcp_dim, self.args.pro_dim)
        self.gcn_x_c_ernie_p = nn.Linear(self.args.ernie_dim, self.args.pro_dim)

        self.cnn_c = nn.Conv2d(in_channels=self.args.channel_num, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

        self.dropout = nn.Dropout(p=self.args.fcDropout)
        self.sigmoid = nn.Sigmoid()


        self.DSC = ResNetDWConv2d(self.args.channel_num)
        self.SKA = SKAttention(self.args.channel_num)



        self.pj_m_doc2vec = nn.Linear(self.args.doc2vec_dim, self.args.embedding_dim)
        self.pj_m_role2vec = nn.Linear(self.args.role2vec_dim, self.args.embedding_dim)
        self.pj_m_tpcp = nn.Linear(self.args.tpcp_dim, self.args.embedding_dim)
        self.pj_m_ernie = nn.Linear(self.args.ernie_dim, self.args.embedding_dim)
        self.pj_m_squence = nn.Linear(self.args.miRNA_numbers, self.args.embedding_dim)

        self.pj_c_doc2vec = nn.Linear(self.args.doc2vec_dim, self.args.embedding_dim)
        self.pj_c_role2vec = nn.Linear(self.args.role2vec_dim, self.args.embedding_dim)
        self.pj_c_tpcp = nn.Linear(self.args.tpcp_dim, self.args.embedding_dim)
        self.pj_c_ernie = nn.Linear(self.args.ernie_dim, self.args.embedding_dim)
        self.pj_c_squence = nn.Linear(self.args.circRNA_numbers, self.args.embedding_dim)

        #############################################
        self.fc1_m = nn.Linear(self.args.embedding_dim * 4, self.args.embedding_dim)
        self.bn1_m = nn.BatchNorm1d(self.args.embedding_dim)
        self.fc2_m = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)
        self.bn2_m = nn.BatchNorm1d(self.args.embedding_dim)
        self.out_m = nn.Sigmoid()

        self.fc1_d = nn.Linear(self.args.embedding_dim * 4, self.args.embedding_dim)
        self.bn1_d = nn.BatchNorm1d(self.args.embedding_dim)
        self.fc2_d = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)
        self.bn2_d = nn.BatchNorm1d(self.args.embedding_dim)
        self.out_d = nn.Sigmoid()

        #############################################
        self.embedding_dim = self.args.embedding_dim

        self.miRNA_projector = MLP_Projector(self.args.circRNA_numbers, self.embedding_dim, self.embedding_dim)
        self.circRNA_projector = MLP_Projector(self.args.miRNA_numbers, self.embedding_dim, self.embedding_dim)

        self.relu1 = nn.ReLU()
        self.fcLinear = MLP(self.args.inSize, 1, dropout=self.args.fcDropout, actFunc=self.relu1)

    def miRNA_embeding(self,  miRNA_doc2vec, miRNA_role2vec, miRNA_tpcp, miRNA_ernie, miRNA_squence):
        miRNA_doc2vec = torch.from_numpy(miRNA_doc2vec).float()
        miRNA_role2vec = torch.from_numpy(miRNA_role2vec).float()
        miRNA_tpcp = torch.from_numpy(miRNA_tpcp).float()
        miRNA_ernie = torch.from_numpy(miRNA_ernie).float()
        miRNA_squence = torch.from_numpy(miRNA_squence).float()

        data = self.mc_graph_dataset
        x_m_f2 = torch.relu(self.gcn_x_m_doc(torch.from_numpy(data['mm_doc']['data_matrix']).float(),
                                             data['mm_doc']['edges']))
        x_m_f2_p = self.dropout(self.gcn_x_m_doc_p(x_m_f2))
        x_m_f3 = torch.relu(self.gcn_x_m_role(torch.from_numpy(data['mm_role']['data_matrix']).float(),
                                              data['mm_role']['edges']))
        x_m_f3_p = self.dropout(self.gcn_x_m_role_p(x_m_f3))
        x_m_f4 = torch.relu(self.gcn_x_m_tpcp(torch.from_numpy(data['mm_tpcp']['data_matrix']).float(),
                                              data['mm_tpcp']['edges']))
        x_m_f4_p = self.dropout(self.gcn_x_m_tpcp_p(x_m_f4))
        x_m_f5 = torch.relu(self.gcn_x_m_ernie(torch.from_numpy(data['mm_ernie']['data_matrix']).float(),
                                               data['mm_ernie']['edges']))
        x_m_f5_p = self.dropout(self.gcn_x_m_ernie_p(x_m_f5))


        m_doc2vec = self.dropout(self.pj_m_doc2vec(miRNA_doc2vec))
        m_role2vec = self.dropout(self.pj_m_role2vec(miRNA_role2vec))
        m_tpcp = self.dropout(self.pj_m_tpcp(miRNA_tpcp))
        m_ernie = self.dropout(self.pj_m_ernie(miRNA_ernie))
        m_squence = self.dropout(self.pj_m_squence(miRNA_squence))


        XM_raw = torch.cat((m_doc2vec, m_role2vec, m_tpcp, m_ernie, m_squence), 1).t()

        x_m = torch.cat((x_m_f2_p, x_m_f3_p, x_m_f4_p, x_m_f5_p), 1)
        XM = XM_raw.view(1, self.args.channel_num, self.args.fm, -1)
        XM_DSC = self.DSC(XM)
        XM_DSC = torch.relu(XM_DSC)
        x = self.cnn_m(XM_DSC)

        x = x.view(self.args.fm, self.args.miRNA_numbers).t()

        return x, x_m

    def circRNA_embeding(self, circRNA_doc2vec,circRNA_role2vec, circRNA_tpcp, circRNA_ernie, circRNA_squence):
        circRNA_doc2vec = torch.from_numpy(circRNA_doc2vec).float()
        circRNA_role2vec = torch.from_numpy(circRNA_role2vec).float()
        circRNA_tpcp = torch.from_numpy(circRNA_tpcp).float()
        circRNA_ernie = torch.from_numpy(circRNA_ernie).float()
        circRNA_squence = torch.from_numpy(circRNA_squence).float()


        data = self.mc_graph_dataset

        x_c_f2 = torch.relu(
            self.gcn_x_c_doc(torch.from_numpy(data['cc_doc']['data_matrix']).float(),
                             data['cc_doc']['edges']))
        x_c_f2_p = self.dropout(self.gcn_x_c_doc_p(x_c_f2))

        x_c_f3 = torch.relu(
             self.gcn_x_c_role(torch.from_numpy(data['cc_role']['data_matrix']).float(),
                               data['cc_role']['edges']))
        x_c_f3_p = self.dropout(self.gcn_x_c_role_p(x_c_f3))
        x_c_f4 = torch.relu(self.gcn_x_c_tpcp(torch.from_numpy(data['cc_tpcp']['data_matrix']).float(),
                                              data['cc_tpcp']['edges']))
        x_c_f4_p = self.dropout(self.gcn_x_c_tpcp_p(x_c_f4))

        x_c_f5 = torch.relu(
            self.gcn_x_c_ernie(torch.from_numpy(data['cc_ernie']['data_matrix']).float(), data['cc_ernie']['edges']))
        x_c_f5_p = self.dropout(self.gcn_x_c_ernie_p(x_c_f5))


        c_doc2vec = self.dropout(self.pj_c_doc2vec(circRNA_doc2vec))
        c_role2vec = self.dropout(self.pj_c_role2vec(circRNA_role2vec))
        c_tpcp = self.dropout(self.pj_c_tpcp(circRNA_tpcp))
        c_ernie = self.dropout(self.pj_c_ernie(circRNA_ernie))
        c_squence = self.dropout(self.pj_c_squence(circRNA_squence))


        XC_raw = torch.cat((c_doc2vec, c_role2vec, c_tpcp, c_ernie, c_squence), 1).t()
        x_c = torch.cat((x_c_f2_p, x_c_f3_p, x_c_f4_p, x_c_f5_p), 1)
        XC = XC_raw.view(1, self.args.channel_num, self.args.fd, -1)  # [1, 3, 128, 2115]

        XC_SKAttention = self.SKA(XC)  # [1, 3, 128, 2115]
        XC_SKAttention = torch.relu(XC_SKAttention)  # [1, 3, 128, 2115]

        y = self.cnn_c(XC_SKAttention)  # [1,1,128,2115]
        y = y.view(self.args.fd, self.args.circRNA_numbers).t()  # [2115, 128]


        return y, x_c

    def ssl_layer_loss(self, context_miRNA_emb_all, context_circRNA_emb_all, initial_miRNA_emb_all,
                       initial_circRNA_emb_all, miRNA, circRNA):
        context_miRNA_emb = context_miRNA_emb_all[miRNA]  # [2048,64]
        initial_miRNA_emb = initial_miRNA_emb_all[miRNA]  # [2048,64]
        norm_miRNA_emb1 = F.normalize(context_miRNA_emb)  # [2048,64]
        norm_miRNA_emb2 = F.normalize(initial_miRNA_emb)  # [2048,64]
        norm_all_miRNA_emb = F.normalize(initial_miRNA_emb_all)  # [410,64]
        pos_score_miRNA = torch.mul(norm_miRNA_emb1, norm_miRNA_emb2).sum(dim=1)  # [2048]
        ttl_score_miRNA = torch.matmul(norm_miRNA_emb1, norm_all_miRNA_emb.transpose(0, 1))  # [2048,410]
        pos_score_miRNA = torch.exp(pos_score_miRNA / self.args.ssl_temp)  # 一个值
        ttl_score_miRNA = torch.exp(ttl_score_miRNA / self.args.ssl_temp).sum(dim=1)  # 一个值
        ssl_loss_miRNA = -torch.log(pos_score_miRNA / ttl_score_miRNA).sum()  # 一个值

        context_circRNA_emb = context_circRNA_emb_all[circRNA]  # [2048,64]
        initial_circRNA_emb = initial_circRNA_emb_all[circRNA]  # [2048,64]
        norm_circRNA_emb1 = F.normalize(context_circRNA_emb)  # [2048,64]
        norm_circRNA_emb2 = F.normalize(initial_circRNA_emb)  # [2048,64]
        norm_all_circRNA_emb = F.normalize(initial_circRNA_emb_all)  # [1931,64]
        pos_score_circRNA = torch.mul(norm_circRNA_emb1, norm_circRNA_emb2).sum(dim=1)  # [2048]
        ttl_score_circRNA = torch.matmul(norm_circRNA_emb1, norm_all_circRNA_emb.transpose(0, 1))  # [2048,1931]
        pos_score_circRNA = torch.exp(pos_score_circRNA / self.args.ssl_temp)  # [2048]
        ttl_score_circRNA = torch.exp(ttl_score_circRNA / self.args.ssl_temp).sum(dim=1)  # [2048]
        ssl_loss_circRNA = -torch.log(pos_score_circRNA / ttl_score_circRNA).sum()  # 一个值

        ssl_loss = self.args.ssl_reg * (ssl_loss_miRNA + self.args.alpha * ssl_loss_circRNA)  # 一个值
        return ssl_loss

    def final_ssl_loss(self, final_miRNA_fea, final_circRNA_fea, m_index, c_index):
        """
        基于最终融合特征的对比学习损失函数 - 双向对比学习

        参数:
        final_miRNA_fea: 最终融合后的miRNA特征 [num_miRNA, dim]
        final_circRNA_fea: 最终融合后的circRNA特征 [num_circRNA, dim]
        m_index: 正样本对中的miRNA索引 [batch_size]
        c_index: 正样本对中的circRNA索引 [batch_size]

        返回:
        ssl_loss: 对比学习损失值
        """
        # ----------- 1. 提取正样本对特征 -----------
        # 根据索引提取当前批次中正样本对的特征
        batch_miRNA_emb = final_miRNA_fea[m_index]  # [batch_size, dim]
        batch_circRNA_emb = final_circRNA_fea[c_index]  # [batch_size, dim]

        # ----------- 2. 特征归一化 -----------
        # 对特征进行L2归一化，便于计算余弦相似度
        norm_batch_miRNA = F.normalize(batch_miRNA_emb, p=2, dim=1)  # [batch_size, dim]
        norm_batch_circRNA = F.normalize(batch_circRNA_emb, p=2, dim=1)  # [batch_size, dim]

        # 对所有特征进行归一化，用于负样本计算
        norm_all_miRNA = F.normalize(final_miRNA_fea, p=2, dim=1)  # [num_miRNA, dim]
        norm_all_circRNA = F.normalize(final_circRNA_fea, p=2, dim=1)  # [num_circRNA, dim]

        # ----------- 3. 双向对比学习 -----------
        tau = self.args.ssl_temp  # 温度系数

        # 视角1：以miRNA为anchor，学习circRNA表示
        # 计算正样本对的相似度得分
        pos_score_miRNA = torch.sum(norm_batch_miRNA * norm_batch_circRNA, dim=1)  # [batch_size]
        # 计算miRNA与所有circRNA的相似度矩阵（用于负样本）
        neg_logits_miRNA = torch.matmul(norm_batch_miRNA, norm_all_circRNA.t())  # [batch_size, num_circRNA]

        # 视角2：以circRNA为anchor，学习miRNA表示
        # 计算正样本对的相似度得分
        pos_score_circRNA = torch.sum(norm_batch_circRNA * norm_batch_miRNA, dim=1)  # [batch_size]
        # 计算circRNA与所有miRNA的相似度矩阵（用于负样本）
        neg_logits_circRNA = torch.matmul(norm_batch_circRNA, norm_all_miRNA.t())  # [batch_size, num_miRNA]

        # ----------- 4. InfoNCE损失计算 -----------
        # miRNA视角的损失计算
        numerator_miRNA = torch.exp(pos_score_miRNA / tau)  # 正样本得分 [batch_size]
        denominator_miRNA = torch.exp(neg_logits_miRNA / tau).sum(dim=1)  # 所有负样本得分之和 [batch_size]
        ssl_loss_miRNA = -torch.log(numerator_miRNA / denominator_miRNA).mean()

        # circRNA视角的损失计算
        numerator_circRNA = torch.exp(pos_score_circRNA / tau)  # 正样本得分 [batch_size]
        denominator_circRNA = torch.exp(neg_logits_circRNA / tau).sum(dim=1)  # 所有负样本得分之和 [batch_size]
        ssl_loss_circRNA = -torch.log(numerator_circRNA / denominator_circRNA).mean()

        # ----------- 5. 总损失 -----------
        # 结合两个视角的损失，加权得到最终对比学习损失
        ssl_loss = self.args.ssl_reg * (ssl_loss_miRNA + ssl_loss_circRNA)

        return ssl_loss

    def forward(self, train_data,  return_feature=None):

        Ed, Ed_cat = self.circRNA_embeding(self.circRNA_doc2vec, self.circRNA_role2vec, self.circRNA_tpcp,
                                                      self.circRNA_ernie,  self.circRNA_squence)

        Em, Em_cat = self.miRNA_embeding(self.miRNA_doc2vec, self.miRNA_role2vec, self.miRNA_tpcp,
                                                   self.miRNA_ernie, self.miRNA_squence)


        Em_cat_project = self.out_m(self.bn2_m(self.fc2_m(self.bn1_m(self.fc1_m(Em_cat)))))
        Ec_cat_project = self.out_d(self.bn2_d(self.fc2_d(self.bn1_d(self.fc1_d(Ed_cat)))))


        edgeData = train_data.t()
        m_index = edgeData[0]
        c_index = edgeData[1]


        ssl_loss_1 = self.ssl_layer_loss(Em_cat_project, Ec_cat_project, Em, Ed, m_index, c_index)


        final_miRNA_fea = torch.cat((Em, Em_cat_project), dim=1)
        final_circRNA_fea = torch.cat((Ed, Ec_cat_project), dim=1)


        ssl_loss_2 = self.final_ssl_loss(final_miRNA_fea, final_circRNA_fea, m_index, c_index)

        mFea, dFea = pro_data(train_data, final_miRNA_fea, final_circRNA_fea)


        node_embed = (mFea.unsqueeze(1) * dFea.unsqueeze(1)).squeeze(dim=1)   # 特征逐元素相乘
        pre_part = self.fcLinear(node_embed)
        # print("node_embed.shape:", node_embed.shape)
        # print("pre_part.shape:", pre_part.shape)  # 应该是 [num_edges, 1]

        pre_asso = self.sigmoid(pre_part).squeeze(dim=1)


        ssl_loss = ssl_loss_1 + ssl_loss_2

        if return_feature:
            return pre_asso, ssl_loss, node_embed
        else:
            return pre_asso, ssl_loss



