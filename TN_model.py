import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_tt import initialize_weights
import numpy as np
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Attention(nn.Module):
    def __init__(self, ):
        super(Attention, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_size = int(256 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义了三个线性层（`Linear`），分别用于处理输入序列的查询（query）、键（key）和值（value）
        self.query = nn.Linear(256, self.all_head_size)  # `self.query`：将输入序列的隐藏层大小映射到所有注意力头的大小
        self.key = nn.Linear(256, self.all_head_size)  # `self.key`：将输入序列的隐藏层大小映射到所有注意力头的大小
        self.value = nn.Linear(256, self.all_head_size)  # `self.value`：将输入序列的隐藏层大小映射到所有注意力头的大小

        self.query_text = nn.Linear(256, self.all_head_size)
        self.key_text = nn.Linear(256, self.all_head_size)
        self.value_text = nn.Linear(256, self.all_head_size)

        self.out = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.25)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # print(self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, img, text=None):
        mixed_query_layer = self.query(img)  # 使用`self.query()`函数将隐藏状态作为查询输入，并获取查询层的结果，命名为`mixed_query_layer`
        mixed_key_layer = self.key(img)  # 使用`self.key()`函数将隐藏状态作为键输入，并获取键层的结果，命名为`mixed_key_layer`
        mixed_value_layer = self.value(img)  # 使用`self.value()`函数将隐藏状态作为值输入，并获取值层的结果，命名为`mixed_value_layer`

        if text is not None:
            t_q = self.query_text(text)
            t_k = self.key_text(text)
            t_v = self.value_text(text)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if text is not None:
            query_layer_img = query_layer
            key_layer_img = key_layer
            value_layer_img = value_layer
            query_layer_text = self.transpose_for_scores(t_q)
            key_layer_text = self.transpose_for_scores(t_k)
            value_layer_text = self.transpose_for_scores(t_v)

        if text is None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))  # 通过torch.matmul计算query_layer与key_layer的转置矩阵相乘得到attention_scores
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 除以sqrt(self.attention_head_size)得到attention_scores
            attention_probs = self.softmax(attention_scores)  # 通过softmax函数对attention_scores进行归一化处理得到attention_probs
            attention_probs = self.dropout(attention_probs)  # 通过self.attn_dropout对attention_probs进行dropout操作

            context_layer = torch.matmul(attention_probs,
                                         value_layer)  # 通过torch.matmul计算attention_probs与value_layer的矩阵乘法得到context_layer
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 通过permute、contiguous和view函数调整其形状
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)  # 通过self.out计算context_layer并返回
            attention_output = self.dropout(attention_output)  # 通过self.proj_dropout对attention_output进行dropout操作并返回
            return attention_output

        else:
            attention_scores_img = torch.matmul(query_layer_img, key_layer_img.transpose(-1, -2))
            attention_scores_text = torch.matmul(query_layer_text, key_layer_text.transpose(-1, -2))
            attention_scores_it = torch.matmul(query_layer_img, key_layer_text.transpose(-1, -2))
            attention_scores_ti = torch.matmul(query_layer_text, key_layer_img.transpose(-1, -2))

            attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)
            attention_probs_img = self.softmax(attention_scores_img)
            attention_probs_img = self.dropout(attention_probs_img)

            # 同上对文本进行操作
            attention_scores_text = attention_scores_text / math.sqrt(self.attention_head_size)
            attention_probs_text = self.softmax(attention_scores_text)
            attention_probs_text = self.dropout(attention_probs_text)

            attention_scores_it = attention_scores_it / math.sqrt(self.attention_head_size)
            attention_probs_it = self.softmax(attention_scores_it)
            attention_probs_it = self.dropout(attention_probs_it)

            attention_scores_ti = attention_scores_ti / math.sqrt(self.attention_head_size)
            attention_probs_ti = self.softmax(attention_scores_ti)
            attention_probs_ti = self.dropout(attention_probs_ti)

            # 通过矩阵乘法计算注意力概率（attention_probs）与值层（value_layer）的乘积，然后对结果进行转置和连续操作，以得到正确的维度。这个过程在图像（img）和文本（text）两种模态上分别进行，最后还计算了图像（img）与文本（text）之间的注意力概率
            context_layer_img = torch.matmul(attention_probs_img, value_layer_img)
            context_layer_img = context_layer_img.permute(0, 2, 1, 3).contiguous()
            context_layer_text = torch.matmul(attention_probs_text, value_layer_text)
            context_layer_text = context_layer_text.permute(0, 2, 1, 3).contiguous()
            context_layer_it = torch.matmul(attention_probs_it, value_layer_text)
            context_layer_it = context_layer_it.permute(0, 2, 1, 3).contiguous()
            context_layer_ti = torch.matmul(attention_probs_ti, value_layer_img)
            context_layer_ti = context_layer_ti.permute(0, 2, 1, 3).contiguous()

            # 用于处理四个不同的张量（context_layer_img、context_layer_text、context_layer_it、context_layer_ti），将它们调整到相同的形状，然后计算两个注意力输出（attention_output_img 和 attention_output_text）
            new_context_layer_shape = context_layer_img.size()[:-2] + (self.all_head_size,)  # 获取context_layer_img的形状，去掉最后一个维度，然后将其与self.all_head_size连接，得到new_context_layer_shape
            context_layer_img = context_layer_img.view(*new_context_layer_shape)  # 将context_layer_img转换为new_context_layer_shape
            new_context_layer_shape = context_layer_text.size()[:-2] + (self.all_head_size,)
            context_layer_text = context_layer_text.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_it.size()[:-2] + (self.all_head_size,)
            context_layer_it = context_layer_it.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ti.size()[:-2] + (self.all_head_size,)
            context_layer_ti = context_layer_ti.view(*new_context_layer_shape)
            attention_output_img = self.out((context_layer_img + context_layer_it) / 2)  # 将context_layer_img和context_layer_it相加，然后除以2，得到attention_output_img
            attention_output_text = self.out((context_layer_text + context_layer_ti) / 2)  # 将context_layer_text和context_layer_ti相加，然后除以2，得到attention_output_text
            attention_output_img = self.dropout(attention_output_img)  # 计算两个注意力输出
            attention_output_text = self.dropout(attention_output_text)

            return attention_output_img, attention_output_text

class Block(nn.Module):
    def __init__(self, mm=False):
        super(Block, self).__init__()
        self.hidden_size = 256     # `hidden_size`：隐藏层的大小
        self.attention_norm = nn.LayerNorm(256, eps=1e-6)       # `attention_norm`：注意力层的归一化层
        self.ffn_norm = nn.LayerNorm(256, eps=1e-6)     # `ffn_norm`：前馈神经网络（FFN）的归一化层
        if mm:      # `att_norm_text`、`ffn_norm_text`、`ffn_text`（仅在`mm`为True时存在）：与文本相关的归一化层和前馈神经网络
            self.att_norm_text = nn.LayerNorm(256, eps=1e-6)
            self.ffn_norm_text = nn.LayerNorm(256, eps=1e-6)
            self.ffn_text = Mlp()

        self.ffn = Mlp()      # `ffn`：前馈神经网络
        self.attn = Attention()      # `attn`：注意力层

    def forward(self, x, text=None):
        # self-attention block
        if text is None:
            h = x
            x = self.attention_norm(x)
            x = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x
        # Bidirectional multimodal attention block
        else:
            h = x
            h_text = text
            x = self.attention_norm(x)
            text = self.att_norm_text(text)
            x, text = self.attn(x, text)
            x = x + h
            text = text + h_text

            h = x
            h_text = text
            x = self.ffn_norm(x)
            text = self.ffn_norm_text(text)
            x = self.ffn(x)
            text = self.ffn_text(text)
            x = x + h
            text = text + h_text
            return x, text


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    """
    初始化了模型的各个组件。
    `self.fc1`是第一个全连接层，输入大小为`config.hidden_size`，输出大小为`config.transformer["mlp_dim"]`。
    `self.fc2`是第二个全连接层，输入大小为`config.transformer["mlp_dim"]`，输出大小为`config.hidden_size`。
    `self.act_fn`是激活函数，这里使用的是GELU激活函数。
    `self.dropout`是一个dropout层，用于在训练过程中随机丢弃部分神经元，以减少过拟合
    """
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.25)

        self._init_weights()

    # `_init_weights`方法用于初始化模型的权重和偏置。这里使用的是Xavier初始化方法，对权重进行均匀分布初始化，对偏置进行正态分布初始化。
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    # `forward`方法定义了模型的前向传播过程。输入`x`首先经过第一个全连接层`self.fc1`，然后经过激活函数`self.act_fn`，再经过dropout层`self.dropout`进行随机丢弃，然后经过第二个全连接层`self.fc2`，最后再经过一次dropout。最终的输出为`x`。
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class MLP_classification(nn.Module):
    def __init__(self):
        super(MLP_classification, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, feature1, feature2):
        # 计算每个特征向量的权重
        weight1 = self.attention(feature1)
        weight2 = self.attention(feature2)

        # 加权融合
        fused_feature = weight1 * feature1 + weight2 * feature2
        return fused_feature

class TN_TOAD(nn.Module):
    def __init__(self, gate=True, size_arg="mid", dropout=False, n_classes=2):
        super(TN_TOAD, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [256, 256, 128], "mid": [1000, 512, 256]}
        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}

        self.attention = Attention().cuda()

        self.relu = nn.ReLU().cuda()
        self.dropout = nn.Dropout(0.25).cuda()

        fc1 = [
            nn.Linear(127, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        ]
        self.func1 = nn.Sequential(*fc1).to(device)

        self.layer = nn.ModuleList().cuda()
        self.linear = nn.Linear(512, 256).cuda()
        for i in range(3):
            # 当`i<2`时，使用带有`mm=True`的`Block`层；否则，使用不带`mm`的`Block`层
            if i < 1:
                layer = Block(mm=True).cuda()
            else:
                layer = Block().cuda()
            self.layer.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(256, eps=1e-6).cuda()

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])

        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc).cuda()

        #self.mcb = MultimodalCompactBilinearPooling(256, 256, 256).cuda()
        #self.attentionfusion = AttentionFusion().cuda()

        self.classifier1= nn.Linear(256, n_classes).cuda()


        self.classifiers = MLP_classification().cuda()

        initialize_weights(self)

        print("-------------------------------", initialize_weights)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        # self.instance_classifiers = self.instance_classifiers.to(device)

    # def forward(self, h, return_features=False,attention_only=False):
    def forward(self, h, gene_data, clinical_data, clinical_age,dna_methylation_data,return_features=False, attention_only=False):

        """feature = torch.cat([clinical_data, clinical_age, gene_data, dna_methylation_data], dim=1)
        feature = self.func1(feature)"""

        feature = torch.cat([clinical_data, clinical_age, gene_data, dna_methylation_data], dim=1)
        feature = self.func1(feature)

        A, h = self.attention_net(h)  # NxKy
        A = torch.transpose(A, 1, 0)  # KxN

        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        M = self.linear(M)
        M = self.relu(M)
        M = self.dropout(M)

        M = M.unsqueeze(0)
        feature = feature.unsqueeze(0)


        for (i, layer_block) in enumerate(self.layer):
            if i == 1:
                hidden_states = torch.cat((M, feature), 1)
                hidden_states = layer_block(hidden_states)
            elif i < 1:
                M, feature = layer_block(M, feature)
            else:
                hidden_states = layer_block(hidden_states)

        hidden_states = self.encoder_norm(hidden_states)
        hidden_states = torch.mean(hidden_states, dim=1)

        """M = self.encoder_norm(M)
        feature = self.encoder_norm(feature)

        feature_all = self.mcb(M, feature)
        feature_all = self.encoder_norm(feature_all)"""

        # hidden_states = torch.cat([M, feature], dim=1)
        #hidden_states = self.attentionfusion(M, feature)

        logits = self.classifier1(hidden_states)
        #logits = self.classifier1(hidden_states)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        #print(Y_prob)
        #print(Y_hat)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})
        return results_dict


