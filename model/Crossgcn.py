# -*- coding:utf-8 -*-
from torch.nn import MultiheadAttention
from lib.utils import scaled_Laplacian, cheb_polynomial
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import sqrt

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)
        # 可视化空间注意力
        # S_normalized_save= S_normalized.cpu().detach().numpy()
        # np.save('S_normalized.npy',S_normalized_save)
        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            # create a full-zero tensor (b,num_of_vertices,out_channels)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                # np.savez('T_k_with_at.npy', T_k_with_at.cpu().detach().numpy())

                theta_k = self.Theta[k]  # (in_channel, out_channel)
                # T_k_with_at * graph_signal
                rhs = T_k_with_at.permute(0, 2, 1).matmul(
                    graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


# class Temporal_Attention_layer(nn.Module):
#     def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
#         super(Temporal_Attention_layer, self).__init__()
#         self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
#         self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
#         self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
#         self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
#         self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))
#
#
#     def forward(self, x):
#         '''
#         :param x: (batch_size, N, F_in, T)
#         :return: (B, T, T)
#         '''
#         _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
#
#         lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
#         # x:(B, N, F_in, T) -> (B, T, F_in, N)
#         # (B, T, F_in, N)(N) -> (B,T,F_in)
#         # (B,T,F_in)(F_in,N)->(B,T,N)
#
#         rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
#
#         product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
#
#         E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
#
#         E_normalized = F.softmax(E, dim=1)
#
#         return E_normalized


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_channels, 1)
        self.multihead_attn2 = nn.MultiheadAttention(num_of_timesteps, 1)
        self.num_of_vertices = num_of_vertices
        self.linear = nn.Linear(in_channels * num_of_timesteps, 512)
        self.linear2 = nn.Linear(512, in_channels * num_of_timesteps)
        self.num_of_timesteps = num_of_timesteps
        self.DEVICE = DEVICE

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        x = x.reshape(_, num_of_vertices, num_of_features * num_of_timesteps)
        x = self.linear(x)
        x = self.linear2(x)
        x = x.reshape(_, num_of_vertices, num_of_features, num_of_timesteps)
        # # 时间注意力
        # x = x.permute(3, 0, 1, 2).reshape(num_of_timesteps, -1, num_of_features)
        # x, _ = self.multihead_attn(x, x, x)
        # x = x.reshape(num_of_timesteps, -1, num_of_vertices, num_of_features).permute(1, 2, 3, 0)
        # 维度注意力
        x = x.permute(2, 0, 1, 3).reshape(num_of_features, -1, num_of_timesteps)
        x, _ = self.multihead_attn2(x, x, x)
        # print(_.shape)
        # np.savez('_.npy', _.cpu().detach().numpy())
        x = x.reshape(num_of_features, -1, num_of_vertices, num_of_timesteps).permute(1, 2, 0, 3)

        return x


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class Crossgcn_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_of_vertices, num_of_timesteps):
        super(Crossgcn_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.routerattn=routerattn(num_of_timesteps,num_of_vertices,1,in_channels*num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上
        self.linear = nn.Linear(in_channels, 512)
        self.linear2 = nn.Linear(512, in_channels)
        self.DEVICE = DEVICE

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        temporal_At = self.TAt(x)

        # routerattn
        temporal_At=temporal_At.permute(0, 2,3,1)

        temporal_At_sat=temporal_At.reshape(batch_size,-1,num_of_vertices)

        routerattn_output,attention_weights = self.routerattn(temporal_At_sat,temporal_At_sat,temporal_At_sat)

        routerattn_output = routerattn_output.reshape(batch_size,num_of_features,num_of_timesteps,num_of_vertices)

        # print(routerattn_output.shape)

        routerattn_output=routerattn_output.permute(0,3,1,2)

        # print(routerattn_output.shape)

        # SAt
        spatial_At = self.SAt(routerattn_output)

        # cheb gcnv
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)



        # convolution along the time axis
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class Crossgcn_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(Crossgcn_submodule, self).__init__()

        self.BlockList = nn.ModuleList([Crossgcn_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                     time_strides, cheb_polynomials, num_of_vertices, len_input)])

        self.BlockList.extend([Crossgcn_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                                            cheb_polynomials, num_of_vertices, len_input // time_strides) for _ in
                               range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict,
               len_input, num_of_vertices):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    # find you a son of bitch
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = Crossgcn_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                             cheb_polynomials, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


class routerattn(nn.Module):
    def __init__(self, seg_num, d_model, n_heads, factor, dropout=0.1):
        super(routerattn, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.seg_num = seg_num
        self.head_dim = d_model // n_heads

        self.router = nn.Parameter(torch.randn(seg_num, factor, self.head_dim))
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dim_sender = nn.Linear(d_model, d_model)
        self.dim_receiver = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)


        batch_router = repeat(self.router, 'seg_num factor head_dim -> b seg_num factor head_dim', b=batch_size)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)


        query = query.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)


        scores = torch.einsum('bnhd,bfhd->bnhf', query, batch_router)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)


        dim_received = torch.einsum('bnhf,bfhd->bnhd', attention_weights, batch_router)

        dim_received = dim_received.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_proj(dim_received)

        return output,attention_weights