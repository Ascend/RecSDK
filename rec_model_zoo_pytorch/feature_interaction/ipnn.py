import sys
import os
import itertools # For generating index pairs

# ==================== 框架上下文代码: START ====================
# 动态添加项目根目录到 Python 路径
try:
    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-2])
    if root_path not in sys.path:
        sys.path.append(root_path)
except NameError:
    if "." not in sys.path:
        sys.path.append(".")

# 导入框架的公共模块和库
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from datasets.criteo import load_data, TestCriteoHandler
from utils.handler import ModelHandler, get_params, get_opts, set_all_seed
# ==================== 框架上下文代码: END ======================


# ============== Main Model: IPNN ==============
class IPNN(nn.Module):
    def __init__(self, params):
        super(IPNN, self).__init__()

        # 1. 加载参数
        self.vocab_size = params.vocab_size # feature_size in tf code
        self.embedding_size = params.embedding_size
        self.field_size = params.field_size
        self.deep_layers_str = params.deep_layers_dnn # Mapped from deep_layers
        self.deep_layers = list(map(int, self.deep_layers_str.split(',')))

        # 2. 定义 Embedding 层
        # 对应 tf code 中的 feat_emb_deep
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # 3. 计算内积层所需的索引
        # Pre-calculate row and column indices for inner product pairs
        row_indices, col_indices = [], []
        for i in range(self.field_size - 1):
            for j in range(i + 1, self.field_size):
                row_indices.append(i)
                col_indices.append(j)
        # Register indices as buffers so they are moved to the correct device
        self.register_buffer('row_indices', torch.tensor(row_indices, dtype=torch.long))
        self.register_buffer('col_indices', torch.tensor(col_indices, dtype=torch.long))
        num_inner_product_pairs = len(row_indices)

        # 4. 定义 Deep 层 (MLP)
        self.dnn_layers = nn.ModuleList()
        # 输入维度 = 展平的 Embedding + 内积结果
        dnn_input_dim = self.field_size * self.embedding_size + num_inner_product_pairs
        for layer_size in self.deep_layers:
            self.dnn_layers.append(nn.Linear(dnn_input_dim, layer_size))
            # TODO: Consider adding BatchNorm and Dropout if needed based on TF defaults
            self.dnn_layers.append(nn.ReLU())
            dnn_input_dim = layer_size # Input for next layer
        # Final output layer (logits)
        self.dnn_output_layer = nn.Linear(dnn_input_dim, 1)

        # 5. 定义损失函数
        self.loss_fn = nn.BCELoss() # Using BCELoss as forward returns probabilities

        # 6. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # Initialize embedding similar to tf.random_normal_initializer(stddev=0.1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        # Initialize linear layers using Xavier uniform (common default for tf.contrib.layers)
        for module in self.dnn_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.dnn_output_layer.weight)
        if self.dnn_output_layer.bias is not None:
             nn.init.zeros_(self.dnn_output_layer.bias)


    def forward(self, features, mode='train'):
        feat_ids = features['feat_ids']      # shape: [batch_size, field_size]
        feat_vals = features['feat_vals']    # shape: [batch_size, field_size]

        # 1. Embedding Lookup and Multiplication
        # shape: [batch_size, field_size, embedding_size]
        embeddings = self.embedding(feat_ids)
        feat_vals = feat_vals.unsqueeze(-1) # shape: [batch_size, field_size, 1]
        embeddings = embeddings * feat_vals

        # 2. Inner Product Layer
        # Gather embeddings for pairs using pre-calculated indices
        p = torch.index_select(embeddings, 1, self.row_indices) # shape: [batch, num_pairs, emb_size]
        q = torch.index_select(embeddings, 1, self.col_indices) # shape: [batch, num_pairs, emb_size]
        # Calculate inner product: sum(p * q) along embedding dimension
        inner_product = torch.sum(p * q, dim=2) # shape: [batch, num_pairs]

        # 3. Deep Layer Input Preparation
        # Flatten embeddings
        emb_for_deep = embeddings.view(-1, self.field_size * self.embedding_size) # shape: [batch, field_size * emb_size]
        # Concatenate flattened embeddings and inner product results
        deep_input = torch.cat([emb_for_deep, inner_product], dim=1)

        # 4. Pass through DNN Layers
        for layer in self.dnn_layers:
            deep_input = layer(deep_input)
        # Get final logits
        logits = self.dnn_output_layer(deep_input)

        # 5. Return prediction probability
        return {'ctr': torch.sigmoid(logits.squeeze(-1))}

    def loss(self, pred, labels):
        """
        损失计算：遵循你的框架接口。
        """
        # Since forward returns probabilities, use BCELoss
        return self.loss_fn(pred['ctr'], labels.float())

if __name__ == '__main__':
    # 1. 获取基础参数
    params = get_params()

    # 2. 更新/设置模型专属和默认参数 (参考 TF 脚本)
    params.update(
    edict({
        'model': 'ipnn', # <-- 指定模型名称
        'vocab_size': 2100000,
        'embedding_size': 10,  # 默认 Embedding 大小
        # field_size 通常由数据加载器决定或在 params 中提供
        'deep_layers_dnn': "256,128,64", # 对应 TF script 中的 layers (默认值示例)
        'learning_rate': 0.001, # 对应 TF script 中的 learning_rate
        # 其他可能需要的参数 (例如 dropout, batch_norm) 可以从 get_params() 获取或在此处设置
    }))

    # 3. 从命令行覆盖参数 
    params = get_opts(sys.argv, params)
    set_all_seed(params)

    # 4. 加载数据 
    # 假设 load_data 会返回正确的 field_size，或者在 get_params 中设置
    if 'field_size' not in params:
         # Set a default or raise an error if field_size is crucial and not provided
         print("Warning: field_size not found in params. Using a default value of 39. Adjust if necessary.")
         params.field_size = 39 # Default Criteo field size, adjust as needed


    train_loader, test_loader, val_loader = load_data(params)


    # 5. 初始化模型
    model = IPNN(params).to(params.device)

    # 6. 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    handler = ModelHandler(params, model, optimizer,
                           load_data,
                           TestCriteoHandler(params))
    handler.run()
