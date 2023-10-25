import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from einops import rearrange
from transformer import Transformer
from alg_parameters import *


def normalized_columns_initializer(weights, std=1.0):
    """weight initializer"""
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """initialize weights"""
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SCRIMPNet(nn.Module):
    """network with transformer-based communication mechanism"""

    def __init__(self):
        """initialization"""
        super(SCRIMPNet, self).__init__()
        self.L = 16
        self.cT = NetParameters.NET_SIZE
        self.mlp_dim = 512
        self.num_classes = 5
        self.heads = 16
        self.depth = 2
        self.emb_dropout = 0.2
        self.transformer_dropout = 0.2

        # observation encoder
        self.conv1 = nn.Conv2d(NetParameters.NUM_CHANNEL, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1a = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.conv1b = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2a = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2b = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE, 3,
                               1, 0)
        self.fully_connected_1 = nn.Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE)
        self.fully_connected_2 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.fully_connected_3 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(8, self.L, 512), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(8, 512, self.cT), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (self.L + 1), self.cT))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cT))
        self.dropout = nn.Dropout(self.emb_dropout)

        self.transformer = Transformer(self.cT, self.depth, self.heads, self.mlp_dim, self.transformer_dropout)

        self.to_cls_token = nn.Identity()

        self.nn_same = nn.Linear(self.cT, self.cT)
        torch.nn.init.xavier_uniform_(self.nn_same.weight)
        torch.nn.init.normal_(self.nn_same.bias, std=1e-6)

        # output heads
        self.policy_layer = nn.Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.sigmoid_layer = nn.Sigmoid()
        self.value_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.blocking_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.apply(weights_init)


    @autocast()
    def forward(self, obs, vector, input_state):
        """run neural network"""
        num_agent = EnvParameters.N_AGENTS
        obs = torch.reshape(obs, (-1, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        vector = torch.reshape(vector, (-1, NetParameters.VECTOR_LEN))
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = F.relu(x_1.view(x_1.size(0), -1))
        # vector input
        x_2 = F.relu(self.fully_connected_1(vector))
        # Concatenation
        x_3 = torch.cat((x_1, x_2), -1)
        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)
        h2 = h2.view(h2.shape[0], h2.shape[1], 1, 1)

        x = rearrange(h2, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,zjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)
        VV = torch.einsum('bij,zjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        # Class tokens and positional embeddings
        cls_tokens = self.cls_token.expand(obs.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Attention
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.nn_same(x)
        x = self.nn_same(x)

        x = torch.reshape(x, (-1, num_agent, NetParameters.NET_SIZE))
        policy_layer = self.policy_layer(x)
        policy = self.softmax_layer(policy_layer)
        policy_sig = self.sigmoid_layer(policy_layer)
        value = self.value_layer(x)
        blocking = F.sigmoid(self.blocking_layer(x))
        return policy, value, blocking, policy_sig, x, policy_layer
