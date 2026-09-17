from typing import Dict, NamedTuple, Optional, List

import torch


class SequentialFeatures:
    """
    存储序列特征的命名元组, 包含历史长度、历史ID、历史嵌入和额外的载荷信息。

    属性：
    - uid: 用户ID哈希值前64位 
    - past_lengths: 历史序列长度, 形状为(B,), 类型为int64, 需要大于0。
    - past_ids: 历史ID, 形状为(B, N), 类型为int64, 0表示有效ID。
    - past_embeddings: 历史Embedding, 形状为(B, N, D), 类型为float, 可以为None。
    - past_payloads: 历史信息, 例如过去的时间戳、事件类型、特征等。
    """

    def __init__(self, uid: torch.Tensor, past_lengths: torch.Tensor, past_ids: torch.Tensor,
                 past_payloads: Dict[str, torch.Tensor], past_embeddings: Optional[torch.Tensor] = None) -> None:
        self.uid = self._assert_not_none(uid, "uid cannot be None")
        self.past_lengths = self._assert_not_none(past_lengths, "past_lengths cannot be None")
        self.past_ids = self._assert_not_none(past_ids, "past_ids cannot be None")
        self.past_payloads = self._assert_not_none(past_payloads, "past_payloads cannot be None")
        self.past_embeddings = past_embeddings

    def to(self, device: torch.device, non_blocking: bool = False) -> "SequentialFeatures":
        self.uid = self.uid.to(device, non_blocking=non_blocking)
        self.past_lengths = self.past_lengths.to(device, non_blocking=non_blocking)
        self.past_ids = self.past_ids.to(device, non_blocking=non_blocking)
        for k in self.past_payloads.keys():
            self.past_payloads[k] = self.past_payloads[k].to(device, non_blocking=non_blocking)
        if self.past_embeddings is not None:
            self.past_embeddings = self.past_embeddings.to(device, non_blocking=non_blocking)
        return self

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self.uid.record_stream(stream)
        self.past_lengths.record_stream(stream)
        self.past_ids.record_stream(stream)
        for k in self.past_payloads.keys():
            self.past_payloads[k].record_stream(stream)
        if self.past_embeddings is not None:
            self.past_embeddings.record_stream(stream)

    def pin_memory(self) -> "SequentialFeatures":
        self.uid = self.uid.pin_memory()
        self.past_lengths = self.past_lengths.pin_memory()
        self.past_ids = self.past_ids.pin_memory()
        for k in self.past_payloads.keys():
            self.past_payloads[k] = self.past_payloads[k].pin_memory()
        if self.past_embeddings is not None:
            self.past_embeddings = self.past_embeddings.pin_memory()
        return self

    @staticmethod
    def _assert_not_none(obj, errmsg):
        if obj is None:
            raise ValueError(errmsg)
        return obj


def seq_features_from_row(
    row: dict,
    device: torch.device,
    max_output_length: int,
    seq_item_feature_names: List[str],
    user_feature_names: List[str],
    include_loss_weights: bool = False,
    include_test_positions: bool = False,
    itemid_column_name='item_id',
    infer_ratings_key='ratings',
    infer_timestamps_key='timestamps',
    include_candidate_items: bool = False
):
    """
    从数据行中提取序列特征。

    :param row: 数据行, 包含历史信息和其他特征。
    :param device: torch tensor所在的device。
    :param max_output_length: 最大输出长度。
    :param seq_item_feature_names: 序列项特征名称列表。
    :param user_feature_names: 用户特征名称列表。
    :param include_loss_weights: 是否包含损失权重, 默认为False。
    :param include_test_positions: 是否包含测试位置, 默认为False, 已废弃。
    :param itemid_column_name: 项目ID列名, 默认为'item_id'。
    :return: SequentialFeatures实例, 包含提取的序列特征。
    """
    historical_lengths = row["history_lengths"].to(device)
    historical_ratings = row[infer_ratings_key].to(device)
    historical_timestamps = row[infer_timestamps_key].to(device)
    uid = row['uid'].to(device)
    batch_size = historical_lengths.size(0)

    if max_output_length > 0:
        historical_ratings = torch.cat([
            historical_ratings,
            torch.zeros((batch_size, max_output_length), dtype=historical_ratings.dtype, device=device),
        ], dim=1)
        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((batch_size, max_output_length), dtype=historical_timestamps.dtype, device=device),
        ], dim=1)

    payload = {infer_ratings_key: historical_ratings, infer_timestamps_key: historical_timestamps}

    for feat_name in seq_item_feature_names:
        feat_ids = row[feat_name].to(device)
        if max_output_length > 0:
            feat_ids = torch.cat([
                feat_ids,
                torch.zeros((batch_size, max_output_length), dtype=feat_ids.dtype, device=device)
            ], dim=1)
        payload[feat_name] = feat_ids

    for feat_name in user_feature_names:
        feat_ids = row[feat_name].to(device)
        payload[feat_name] = feat_ids

    features = SequentialFeatures(
        uid=uid,
        past_lengths=historical_lengths,
        past_ids=payload.get(itemid_column_name),
        past_embeddings=None,
        past_payloads=payload,
    )

    if include_loss_weights:
        loss_weight = row['loss_weights'].to(device)
        labels = row['labels'].to(device)
        if max_output_length > 0:
            loss_weight = torch.cat([
                loss_weight,
                torch.zeros((batch_size, max_output_length), dtype=loss_weight.dtype, device=device),
            ], dim=1)
            labels = torch.cat([
                labels,
                torch.zeros((batch_size, max_output_length), dtype=labels.dtype, device=device),
            ], dim=1)
        features.past_payloads['loss_weights'] = loss_weight
        features.past_payloads['labels'] = labels

    if include_candidate_items:
        for feat_name in seq_item_feature_names:
            features.past_payloads['candidate_' + feat_name] = row['candidate_' + feat_name].to(device)
        features.past_payloads['candidate_item_id'] = row[f'candidate_{itemid_column_name}'].to(device)
        features.past_payloads['candidate_timestamps'] = row['candidate_timestamps'].to(device)
        features.past_payloads['candidate_labels'] = row['candidate_labels'].to(device)
        return features

    return features


def seq_features_from_row_cpu(
    row: dict,
    max_output_length: int,
    seq_item_feature_names: List[str],
    user_feature_names: List[str],
    include_loss_weights: bool = False,
    include_test_positions: bool = False,
    itemid_column_name='item_id',
    infer_ratings_key='ratings',
    infer_timestamps_key='timestamps',
    include_candidate_items: bool = False
):
    """
    从数据行中提取序列特征。

    :param row: 数据行, 包含历史信息和其他特征。
    :param max_output_length: 最大输出长度。
    :param seq_item_feature_names: 序列项特征名称列表。
    :param user_feature_names: 用户特征名称列表。
    :param include_loss_weights: 是否包含损失权重, 默认为False。
    :param include_test_positions: 是否包含测试位置, 默认为False。
    :param itemid_column_name: 项目ID列名, 默认为'item_id'。
    :return: SequentialFeatures实例, 包含提取的序列特征。
    """
    historical_lengths = row["history_lengths"]
    historical_ratings = row[infer_ratings_key]
    historical_timestamps = row[infer_timestamps_key]
    uid = row['uid']
    batch_size = historical_lengths.size(0)

    if max_output_length > 0:
        historical_ratings = torch.cat([
            historical_ratings,
            torch.zeros((batch_size, max_output_length), dtype=historical_ratings.dtype),
        ], dim=1)
        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((batch_size, max_output_length), dtype=historical_timestamps.dtype),
        ], dim=1)

    payload = {infer_ratings_key: historical_ratings, infer_timestamps_key: historical_timestamps}

    for feat_name in seq_item_feature_names:
        feat_ids = row[feat_name]
        if max_output_length > 0:
            feat_ids = torch.cat([
                feat_ids,
                torch.zeros((batch_size, max_output_length), dtype=feat_ids.dtype)
            ], dim=1)
        payload[feat_name] = feat_ids

    for feat_name in user_feature_names:
        feat_ids = row[feat_name]
        payload[feat_name] = feat_ids

    features = SequentialFeatures(
        uid=uid,
        past_lengths=historical_lengths,
        past_ids=payload.get(itemid_column_name),
        past_embeddings=None,
        past_payloads=payload,
    )

    if include_loss_weights:
        loss_weights = row['loss_weights']
        labels = row['labels']
        if max_output_length > 0:
            loss_weights = torch.cat([
                loss_weights,
                torch.zeros((batch_size, max_output_length), dtype=row['loss_weights'].dtype),
            ], dim=1)
            labels = torch.cat([
                labels,
                torch.zeros((batch_size, max_output_length), dtype=row['labels'].dtype),
            ], dim=1)
        features.past_payloads['loss_weights'] = loss_weights
        features.past_payloads['labels'] = labels

    if include_candidate_items:
        for feat_name in seq_item_feature_names:
            features.past_payloads['candidate_' + feat_name] = row['candidate_' + feat_name]
        features.past_payloads['candidate_item_id'] = row[f'candidate_{itemid_column_name}']
        features.past_payloads['candidate_timestamps'] = row['candidate_timestamps']
        features.past_payloads['candidate_labels'] = row['candidate_labels']
    return features