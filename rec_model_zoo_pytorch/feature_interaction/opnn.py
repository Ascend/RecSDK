import sys
import os
import itertools

# ==================== 框架上下文代码: START ====================
try:
    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
    if root_path not in sys.path:
        sys.path.append(root_path)
except NameError:
    if "." not in sys.path:
        sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
# ==================== 框架上下文代码: END ======================


# ============== Main Model: OPNN (TF Style Outer Product) ==============
class OPNN(nn.Module):
    def __init__(self, params):
        super(OPNN, self).__init__()

        # --- 模型参数 ---
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.field_size = params.field_size
        self.deep_layers_str = params.deep_layers_dnn
        self.deep_layers = list(map(int, self.deep_layers_str.split(',')))

        # --- 模型层 ---
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # 预计算外积对的索引
        row_indices, col_indices = [], []
        for i in range(self.field_size - 1):
            for j in range(i + 1, self.field_size):
                row_indices.append(i)
                col_indices.append(j)
        self.register_buffer('row_indices', torch.tensor(row_indices, dtype=torch.long))
        self.register_buffer('col_indices', torch.tensor(col_indices, dtype=torch.long))
        num_pairs = len(row_indices)

        # 外积核 (Kernel)
        # TF shape: [emb_size, num_pairs, emb_size]
        self.outer_product_kernel = nn.Parameter(torch.Tensor(self.embedding_size, num_pairs, self.embedding_size))

        # DNN 层
        self.dnn_layers = nn.ModuleList()
        dnn_input_dim = self.field_size * self.embedding_size + num_pairs
        for layer_size in self.deep_layers:
            self.dnn_layers.append(nn.Linear(dnn_input_dim, layer_size))
            self.dnn_layers.append(nn.ReLU()) 
            dnn_input_dim = layer_size
        self.dnn_output_layer = nn.Linear(dnn_input_dim, 1)

        # --- 损失函数 ---
        self.loss_fn = nn.BCELoss()

        # --- 初始化权重 ---
        self._init_weights()

    def _init_weights(self):
        """初始化权重 (模仿 TF 实现)"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.outer_product_kernel, mean=0.0, std=0.1)
        for module in self.modules():
             if isinstance(module, nn.Linear):
                 nn.init.xavier_uniform_(module.weight)
                 if module.bias is not None:
                     nn.init.zeros_(module.bias)

    def forward(self, features, mode='train'):
        feat_ids = features['feat_ids']
        feat_vals = features['feat_vals'].unsqueeze(-1)

        # --- Embedding ---
        embeddings = self.embedding(feat_ids) * feat_vals # [batch, field, emb]

        # --- Outer Product (TF Logic Simulation) ---
        p = torch.index_select(embeddings, 1, self.row_indices) # [batch, num_pairs, emb]
        q = torch.index_select(embeddings, 1, self.col_indices) # [batch, num_pairs, emb]

        # 1. tmp = tf.expand_dims(p, axis=1) * kernel
        #    p shape: b x N x E   kernel shape: E x N x E
        #    Expand p: b x 1 x N x E
        #    Kernel shape E x N x E -> 1 x E x N x E (for broadcasting)
        p_expanded = p.unsqueeze(1)                   # Shape: b x 1 x N x E
        kernel_expanded = self.outer_product_kernel.unsqueeze(0) # Shape: 1 x E x N x E
        # Element-wise multiply with broadcasting -> (b x E x N x E)
        tmp_step1 = p_expanded * kernel_expanded     # Shape: b x E x N x E

        # 2. tmp = tf.reduce_sum(tmp, axis=-1)
        #    Sum over the last dimension (embedding size)
        tmp_step2 = torch.sum(tmp_step1, dim=-1)     # Shape: b x E x N

        # 3. tmp = tf.transpose(tmp, perm=(0, 2, 1))
        #    Transpose dimensions 1 and 2
        tmp_step3 = tmp_step2.permute(0, 2, 1)       # Shape: b x N x E

        # 4. outer_product = tf.reduce_sum(tmp * q, axis=-1)
        #    Element-wise multiply tmp (b x N x E) with q (b x N x E)
        #    Then sum over the last dimension (embedding size)
        outer_product = torch.sum(tmp_step3 * q, dim=-1) # Shape: b x N (num_pairs)
        # --- End Outer Product ---

        # --- DNN 输入 ---
        emb_for_deep = embeddings.view(-1, self.field_size * self.embedding_size)
        deep_input = torch.cat([emb_for_deep, outer_product], dim=1)

        # --- DNN 前向传播 ---
        for layer in self.dnn_layers:
            deep_input = layer(deep_input)
        logits = self.dnn_output_layer(deep_input)

        # --- 输出 ---
        return {'ctr': torch.sigmoid(logits.squeeze(-1))}

    def loss(self, pred, labels):
        """计算损失"""
        return self.loss_fn(pred['ctr'], labels.float())

# ============== 主执行函数 ==============
if __name__ == '__main__':
    params = get_params()
    params.update(
    edict({
        'model': 'opnn', # 模型名保持 opnn
        'vocab_size': 2100000,
        'embedding_size': 10,
        'deep_layers_dnn': "256,128,64",
        'learning_rate': 0.001,
    }))
    params = get_opts(sys.argv, params)
    set_all_seed(params)

    if 'field_size' not in params:
         print("Warning: field_size not defined. Using default 39.")
         params.field_size = 39

    train_loader, test_loader, val_loader = load_data(params)
    model = OPNN(params).to(params.device) # 实例化主 OPNN 类
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    handler = ModelHandler(params, model, optimizer,
                           load_data, TestCriteoHandler(params))
    handler.run()
