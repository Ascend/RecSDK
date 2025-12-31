import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeneralSearchUnit(nn.Module):
    """
    通用搜索单元，用于从用户行为序列中检索与目标物品相关的子序列
    支持软搜索（soft）和硬搜索（hard）两种模式

    Args:
        item_embedding_dim (int): 物品嵌入维度
        hidden_dim (int, optional): 隐层维度，默认128
        search_type (str, optional): 搜索类型，可选"soft"或"hard"，默认"hard"
        num_categories (int, optional): 物品类别数量，仅在硬搜索时使用
    """
    def __init__(self, item_embedding_dim, hidden_dim=128, search_type="hard", num_categories=None):
        super().__init__()
        self.search_type = search_type
        self.item_embedding_dim = item_embedding_dim

        if search_type == "soft":
            self.W_b = nn.Linear(item_embedding_dim, hidden_dim)
            self.W_a = nn.Linear(item_embedding_dim, hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif search_type == "hard":
            self.num_categories = num_categories

    def forward(self, user_behavior_seq, target_item, behavior_categories=None, target_category=None, k=100):
        """
        前向传播方法

        Args:
            user_behavior_seq (Tensor): 用户行为序列，形状为[batch_size, seq_len, embedding_dim]
            target_item (Tensor): 目标物品嵌入，形状为[batch_size, embedding_dim]
            behavior_categories (Tensor, optional): 用户行为类别，仅在硬搜索时使用
            target_category (Tensor, optional): 目标物品类别，仅在硬搜索时使用
            k (int, optional): 返回的top-k数量，默认100

        Returns:
            tuple: 包含三个元素的元组，分别为
                - top_k_behavior (Tensor): top-k行为序列
                - top_k_indices (Tensor): top-k行为索引
                - top_k_scores (Tensor): top-k行为得分
        """
        if self.search_type == "soft":
            return self._soft_search(user_behavior_seq, target_item, k)
        elif self.search_type == "hard":
            return self._hard_search(user_behavior_seq, behavior_categories, target_category, k)

    def _soft_search(self, user_behavior_seq, target_item, k):
        """
        软搜索方法，通过计算相似度得分来选择top-k行为

        Args:
            user_behavior_seq (Tensor): 用户行为序列
            target_item (Tensor): 目标物品嵌入
            k (int): 返回的top-k数量

        Returns:
            tuple: 包含三个元素的元组，分别为
                - top_k_behavior (Tensor): top-k行为序列
                - top_k_indices (Tensor): top-k行为索引
                - top_k_scores (Tensor): top-k行为得分
        """
        batch_size, seq_len, embedding_dim = user_behavior_seq.size()

        behavior_transformed = self.W_b(user_behavior_seq)
        target_transformed = self.W_a(target_item).unsqueeze(1)

        # 防止除零的微小数值
        EPSILON = 1e-8

        # 计算余弦相似度
        behavior_norm = torch.norm(behavior_transformed, p=2, dim=-1, keepdim=True)
        target_norm = torch.norm(target_transformed, p=2, dim=-1, keepdim=True)
        similarity = torch.sum(behavior_transformed * target_transformed, dim=-1) / (behavior_norm * target_norm + EPSILON)

        relevance_scores = self.mlp(similarity.unsqueeze(-1)).squeeze(-1)

        top_k_scores, top_k_indices = torch.topk(relevance_scores, k=min(k, seq_len), dim=-1)

        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        top_k_behavior = user_behavior_seq[batch_indices, top_k_indices]

        return top_k_behavior, top_k_indices, top_k_scores

    def _hard_search(self, user_behavior_seq, behavior_categories, target_category, k):
        """
        硬搜索方法，通过类别匹配选择行为

        Args:
            user_behavior_seq (Tensor): 用户行为序列
            behavior_categories (Tensor): 用户行为类别
            target_category (Tensor): 目标物品类别
            k (int): 返回的top-k数量

        Returns:
            tuple: 包含三个元素的元组，分别为
                - matched_behaviors (Tensor): 匹配的行为序列
                - matched_indices (Tensor): 匹配的行为索引
                - matched_scores (Tensor): 匹配的行为得分
        """
        batch_size, seq_len, embedding_dim = user_behavior_seq.size()
        device = user_behavior_seq.device

        matched_behaviors = []
        matched_indices = []
        matched_scores = []

        for i in range(batch_size):
            if target_category is not None:
                target_cat = target_category[i].item()
                cat_mask = (behavior_categories[i] == target_cat).nonzero(as_tuple=False).squeeze(-1)
            else:
                cat_mask = torch.arange(min(k, seq_len), device=device)

            if len(cat_mask) == 0:
                cat_mask = torch.arange(min(k, seq_len), device=device)

            if len(cat_mask) > k:
                cat_mask = cat_mask[:k]

            matched_behavior = user_behavior_seq[i, cat_mask]

            if len(cat_mask) < k:
                pad_len = k - len(cat_mask)
                pad_behavior = torch.zeros(pad_len, embedding_dim, device=device)
                matched_behavior = torch.cat([matched_behavior, pad_behavior], dim=0)
                pad_indices = torch.full((pad_len,), -1, dtype=torch.long, device=device)
                cat_mask = torch.cat([cat_mask, pad_indices], dim=0)

            matched_behaviors.append(matched_behavior)
            matched_indices.append(cat_mask)

            scores = torch.zeros(seq_len, device=device)
            if target_category is not None:
                scores[behavior_categories[i] == target_cat] = 1.0
            matched_scores.append(scores)

        matched_behaviors = torch.stack(matched_behaviors)
        matched_indices = torch.stack(matched_indices)
        matched_scores = torch.stack(matched_scores)

        top_k_scores, _ = torch.topk(matched_scores, k=min(k, seq_len), dim=-1)

        return matched_behaviors, matched_indices, top_k_scores

class ExactSearchUnit(nn.Module):
    """
    精确搜索单元，用于通过注意力机制聚合用户兴趣

    Args:
        embedding_dim (int): 嵌入维度
        num_heads (int, optional): 多头注意力头数，默认8
        hidden_dim (int, optional): 隐层维度，默认256
        dropout (float, optional): dropout概率，默认0.1
    """
    def __init__(self, embedding_dim, num_heads=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 500 个时间步（0~499），用于时间嵌入。后续若需支持更长时间，可调整此值
        self.time_embedding = nn.Embedding(500, embedding_dim)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, sub_behavior_seq, target_item_emb, time_intervals):
        """
        前向传播方法

        Args:
            sub_behavior_seq (Tensor): 子行为序列，形状为[batch_size, k, embedding_dim]
            target_item_emb (Tensor): 目标物品嵌入，形状为[batch_size, embedding_dim]
            time_intervals (Tensor): 时间间隔信息

        Returns:
            tuple: 包含两个元素的元组，分别为
                - user_interest (Tensor): 用户兴趣表示
                - attention_weights (Tensor): 注意力权重
        """
        batch_size, k, _ = sub_behavior_seq.size()
        time_emb = self.time_embedding(time_intervals)
        behavior_with_time = sub_behavior_seq + time_emb

        q = self.q_proj(behavior_with_time)
        k_proj = self.k_proj(behavior_with_time)
        v = self.v_proj(behavior_with_time)

        q = q.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.einsum('bhik,bhjk->bhij', q, k_proj) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.einsum('bhij,bhjk->bhik', attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, k, self.embedding_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm1(attn_output + behavior_with_time)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.norm2(ffn_output + attn_output)

        target_expanded = target_item_emb.unsqueeze(1).expand(-1, k, -1)
        attention_scores = torch.sum(ffn_output * target_expanded, dim=-1, keepdim=True)
        attention_weights = F.softmax(attention_scores, dim=1)
        user_interest = torch.sum(ffn_output * attention_weights, dim=1)
        output = self.output_layer(user_interest)

        return output, attention_weights.squeeze(-1)

class SIMModel(nn.Module):
    """
    SIM模型，结合通用搜索单元和精确搜索单元进行点击率预测

    Args:
        item_embedding_dim (int): 物品嵌入维度
        user_feature_dim (int, optional): 用户特征维度，默认32
        hidden_dim (int, optional): 隐层维度，默认128
        num_heads (int, optional): 多头注意力头数，默认8
        dropout (float, optional): dropout概率，默认0.1
        search_type (str, optional): 搜索类型，可选"soft"或"hard"，默认"hard"
        num_categories (int, optional): 物品类别数量，仅在硬搜索时使用
    """
    def __init__(self,
                item_embedding_dim,
                user_feature_dim=32,
                hidden_dim=128,
                num_heads=8,
                dropout=0.1,
                search_type="hard",
                num_categories=None):
        super().__init__()
        self.item_embedding_dim = item_embedding_dim
        self.user_feature_dim = user_feature_dim

        self.gsu = GeneralSearchUnit(
            item_embedding_dim=item_embedding_dim,
            hidden_dim=hidden_dim,
            search_type=search_type,
            num_categories=num_categories,
        )

        self.esu = ExactSearchUnit(
            embedding_dim=item_embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(item_embedding_dim * 2 + user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,
                user_behavior_seq,
                target_item_emb,
                user_features=None,
                behavior_categories=None,
                target_category=None,
                time_intervals=None):
        """
        前向传播方法

        Args:
            user_behavior_seq (Tensor): 用户行为序列
            target_item_emb (Tensor): 目标物品嵌入
            user_features (Tensor, optional): 用户特征
            behavior_categories (Tensor, optional): 用户行为类别
            target_category (Tensor, optional): 目标物品类别
            time_intervals (Tensor, optional): 时间间隔信息

        Returns:
            tuple: 包含三个元素的元组，分别为
                - ctr_prediction (Tensor): 点击率预测结果
                - attention_weights (Tensor): 注意力权重
                - indices (Tensor): 选择的行为索引
        """
        batch_size, seq_len, embedding_dim = user_behavior_seq.size()

        if self.gsu.search_type == "hard":
            sub_behavior_seq, indices, scores = self.gsu(
                user_behavior_seq,
                target_item_emb,
                behavior_categories,
                target_category,
                k=min(100, seq_len), #最大候选行为数量，默认为100
            )
        else:
            sub_behavior_seq, indices, scores = self.gsu(
                user_behavior_seq, target_item_emb,k=min(100, seq_len) #最大候选行为数量，默认为100
            )

        if time_intervals is None:
            k = sub_behavior_seq.size(1)
            time_intervals = torch.arange(k, device=user_behavior_seq.device).unsqueeze(0).expand(batch_size, -1)
            time_intervals = torch.clamp(time_intervals, 0, 499)
        else:
            k = sub_behavior_seq.size(1)
            if time_intervals.size(1) != k:
                if time_intervals.size(1) > k:
                    time_intervals = time_intervals[:, :k]
                else:
                    pad_size = k - time_intervals.size(1)
                    pad_values = torch.zeros(batch_size, pad_size, dtype=time_intervals.dtype, device=time_intervals.device)
                    time_intervals = torch.cat([time_intervals, pad_values], dim=1)

        user_interest, attention_weights = self.esu(sub_behavior_seq, target_item_emb, time_intervals)

        if user_features is not None:
            combined_features = torch.cat([user_interest, target_item_emb, user_features], dim=-1)
        else:
            combined_features = torch.cat([user_interest, target_item_emb], dim=-1)
            dummy_features = torch.zeros(batch_size, self.user_feature_dim, device=user_interest.device)
            combined_features = torch.cat([combined_features, dummy_features], dim=-1)

        ctr_prediction = self.predict_layer(combined_features)

        return ctr_prediction, attention_weights, indices

class SIMLoss(nn.Module):
    def __init__(self, search_type="hard", loss_weights=None):
        super().__init__()
        self.search_type = search_type
        self.loss_weights = loss_weights or {'alpha': 1.0, 'beta': 1.0}
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_ctr, target_ctr, gsu_pred=None, gsu_target=None):
        esu_loss = self.bce_loss(pred_ctr.squeeze(), target_ctr.squeeze())

        gsu_loss = 0.0
        if self.search_type == "soft" and gsu_pred is not None and gsu_target is not None:
            gsu_loss = self.bce_loss(gsu_pred, gsu_target)

        total_loss = self.loss_weights['alpha'] * gsu_loss + self.loss_weights['beta'] * esu_loss

        return total_loss
