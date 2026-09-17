import os

import torch


def _is_npu_run() -> bool:
    return os.environ.get("NPU_FLAG", "True") != "False"


def _split_files_by_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset
    ds.files = ds.files[worker_id::worker_info.num_workers]
    ds.init_multi_csv_iterator()


def _base_loader_kwargs(num_workers, prefetch_factor, pin_memory=False, persistent_workers=False):
    num_workers = max(0, int(num_workers))
    kwargs = {
        "num_workers": num_workers,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["worker_init_fn"] = _split_files_by_worker
        kwargs["persistent_workers"] = bool(persistent_workers)
        if _is_npu_run():
            start_method = os.environ.get("NPU_DATALOADER_START_METHOD", "spawn").strip()
            if start_method and start_method.lower() != "default":
                kwargs["multiprocessing_context"] = start_method
    if pin_memory:
        kwargs["pin_memory"] = (not _is_npu_run()) or os.environ.get("NPU_DATALOADER_PIN_MEMORY", "0") in ("1", "true", "True")
    return kwargs


class _SequentialEPCollator:
    def __init__(
            self,
            item_feature_columns,
            user_feature_columns,
            max_output_length=0,
            itemid_column_name='item_id',
            infer_ratings_key='ratings',
            infer_timestamps_key='timestamps',
            multi_value_prefix='pref_',
            include_loss_weights=False,
            include_candidate_items=False,
    ):
        self.item_feature_columns = item_feature_columns
        self.user_feature_columns = user_feature_columns
        self.max_output_length = max_output_length
        self.itemid_column_name = itemid_column_name
        self.infer_ratings_key = infer_ratings_key
        self.infer_timestamps_key = infer_timestamps_key
        self.multi_value_prefix = multi_value_prefix
        self.include_loss_weights = include_loss_weights
        self.include_candidate_items = include_candidate_items

    def __call__(self, data):
        from modeling.generic.sequential_v2.features import SequentialFeatures
        from torchrec import KeyedJaggedTensor, JaggedTensor
        from modeling.generic.sequential_v2.torchrec_trainer import Batch

        uids, past_lengths, past_payloads = [], [], {}
        max_output_length = self.max_output_length
        itemid_column_name = self.itemid_column_name
        infer_ratings_key = self.infer_ratings_key
        infer_timestamps_key = self.infer_timestamps_key

        for row in data:
            past_lengths.append(row["history_lengths"])
            uids.append(row['uid'])
            historical_ratings = row[infer_ratings_key]
            historical_timestamps = row[infer_timestamps_key]

            if max_output_length > 0:
                historical_ratings = torch.cat([
                    historical_ratings,
                    torch.zeros(max_output_length, dtype=historical_ratings.dtype),
                ])
                historical_timestamps = torch.cat([
                    historical_timestamps,
                    torch.zeros(max_output_length, dtype=historical_timestamps.dtype),
                ])

            past_payloads.setdefault(infer_ratings_key, []).append(historical_ratings.unsqueeze(0))
            past_payloads.setdefault(infer_timestamps_key, []).append(historical_timestamps.unsqueeze(0))

            for feat_name, _ in self.item_feature_columns.items():
                feat_ids = row[feat_name]
                if max_output_length > 0:
                    feat_ids = torch.cat([
                        feat_ids,
                        torch.zeros(max_output_length, dtype=feat_ids.dtype)
                    ])
                past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

            for feat_name, _ in self.user_feature_columns.items():
                feat_ids = row[feat_name]
                past_payloads.setdefault(feat_name, []).append(feat_ids.unsqueeze(0))

            if self.include_loss_weights:
                labels = row['labels']
                loss_weights = row['loss_weights']
                if max_output_length > 0:
                    loss_weights = torch.cat([
                        loss_weights,
                        torch.zeros(max_output_length, dtype=row['loss_weights'].dtype),
                    ])
                    labels = torch.cat([
                        labels,
                        torch.zeros(max_output_length, dtype=row['labels'].dtype),
                    ])
                past_payloads.setdefault('loss_weights', []).append(loss_weights.unsqueeze(0))
                past_payloads.setdefault('labels', []).append(labels.unsqueeze(0))

            if self.include_candidate_items:
                for feat_name, feat_cfg in self.item_feature_columns.items():
                    candidate_feat_name = 'candidate_' + feat_name
                    past_payloads.setdefault(candidate_feat_name, []).append(row[candidate_feat_name].unsqueeze(0))
                past_payloads.setdefault('candidate_item_id', []).append(row[f'candidate_{itemid_column_name}'].unsqueeze(0))
                past_payloads.setdefault('candidate_timestamps', []).append(row['candidate_timestamps'].unsqueeze(0))
                past_payloads.setdefault('candidate_labels', []).append(row['candidate_labels'].unsqueeze(0))

        uids = torch.LongTensor(uids)
        past_lengths = torch.LongTensor(past_lengths)

        for key, lst in past_payloads.items():
            if len(lst) > 0 and isinstance(lst[0], torch.Tensor) and lst[0].dim() > 0:
                past_payloads[key] = torch.concat(lst)
            else:
                past_payloads[key] = torch.tensor(lst)
        past_ids = past_payloads.get(itemid_column_name).clone()

        item_kjt_dict = {}
        if self.include_candidate_items:
            for feat_name, feat_cfg in self.item_feature_columns.items():
                if feat_cfg.get("enabled", True):
                    candidate_feat_name = 'candidate_' + feat_name
                    item_feature_id = past_payloads[feat_name].reshape(-1)
                    candidate_feature_id = past_payloads[candidate_feat_name].reshape(-1)
                    feature_id = torch.cat([item_feature_id, candidate_feature_id])
                    jagged_tensor = JaggedTensor(values=feature_id.long(),
                                                 lengths=torch.ones_like(feature_id, dtype=torch.int64))
                    item_kjt_dict[feat_name] = jagged_tensor
        else:
            for feat_name, feat_cfg in self.item_feature_columns.items():
                if feat_cfg.get("enabled", True):
                    feature_id = past_payloads[feat_name].reshape(-1)
                    jagged_tensor = JaggedTensor(values=feature_id.long(),
                                                 lengths=torch.ones_like(feature_id, dtype=torch.int64))
                    item_kjt_dict[feat_name] = jagged_tensor

        user_kjt_dict, user_asso_pref_feature_names = {}, {}
        for feat_name, feat_cfg in self.user_feature_columns.items():
            if feat_cfg.get("enabled", True):
                feat_type = feat_cfg.get("dtype", "int")
                if feat_type == "int":
                    feature_id = past_payloads[feat_name].reshape(-1)
                    jagged_tensor = JaggedTensor(values=feature_id.long(),
                                                 lengths=torch.ones_like(feature_id, dtype=torch.int64))
                    user_kjt_dict[feat_name] = jagged_tensor
                elif feat_type == "pref":
                    cfg_asso_name = feat_cfg.get("associated", None)
                    associated_name = feat_name.replace(self.multi_value_prefix, "") if cfg_asso_name is None else cfg_asso_name
                    if associated_name in item_kjt_dict:
                        user_asso_pref_feature_names.setdefault(associated_name, []).append(feat_name)

        shared_item_kjt_dicts = {}
        for asso_name, pref_names in user_asso_pref_feature_names.items():
            kjt_key = f"item_ebc_shared{len(pref_names)}"
            feature_id_list = [item_kjt_dict.pop(asso_name).values()]
            for pref_name in pref_names:
                feature_id_list.append(past_payloads[pref_name].reshape(-1))
            feature_id = torch.cat(feature_id_list)
            jagged_tensor = JaggedTensor(values=feature_id.long(),
                                         lengths=torch.ones_like(feature_id, dtype=torch.int64))
            shared_item_kjt_dicts.setdefault(kjt_key, {})[asso_name] = jagged_tensor

        if shared_item_kjt_dicts:
            for kjt_key, shared_item_kjt_dict in shared_item_kjt_dicts.items():
                past_payloads[kjt_key] = KeyedJaggedTensor.from_jt_dict(shared_item_kjt_dict)

        item_kjt = KeyedJaggedTensor.from_jt_dict(item_kjt_dict)
        past_payloads["item_kjt"] = item_kjt

        user_kjt = KeyedJaggedTensor.from_jt_dict(user_kjt_dict)
        past_payloads["user_kjt"] = user_kjt

        features = SequentialFeatures(
            uid=uids,
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=None,
            past_payloads=past_payloads
        )
        return Batch(features)


def create_data_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 加载数据时使用的子进程数量, 默认为CPU核心数。
    :return: 一个配置好的DataLoader对象。
    """

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_base_loader_kwargs(num_workers, prefetch_factor)
    )

    return data_loader


def create_data_loader_ep(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        item_feature_columns,
        user_feature_columns,
        max_output_length=0,
        itemid_column_name='item_id',
        infer_ratings_key='ratings',
        infer_timestamps_key='timestamps',
        multi_value_prefix='pref_',
        include_loss_weights: bool = False,
        include_candidate_items: bool = False,
        prefetch_factor: int = 128,
        num_workers: int = os.cpu_count(),
) -> torch.utils.data.DataLoader:
    """
    创建一个数据加载器(DataLoader), 用于批量加载数据集。

    :param dataset: 要加载的数据集。
    :param batch_size: 每个批次的样本数量。
    :param item_feature_columns: 商品特征信息。
    :param user_feature_columns: 用户特征信息。
    :param max_output_length: 最大输出长度。
    :param itemid_column_name: 项目ID列名, 默认为'item_id'。
    :param infer_ratings_key: 项目rating列名, 默认为'ratings'。
    :param infer_timestamps_key: 项目timestamp列名, 默认为'timestamps'。
    :param multi_value_prefix: 用户关联商品特征的特征前缀。
    :param include_loss_weights: 是否包含loss权重。
    :param include_candidate_items: 是否包含候选特征，评估阶段为True。
    :param prefetch_factor: 预取因子，用于控制预取的数据量。
    :param num_workers: 加载数据时使用的子进程数量, 默认为CPU核心数。
    :return: 一个配置好的DataLoader对象。
    """
    collect_fn = _SequentialEPCollator(
        item_feature_columns=item_feature_columns,
        user_feature_columns=user_feature_columns,
        max_output_length=max_output_length,
        itemid_column_name=itemid_column_name,
        infer_ratings_key=infer_ratings_key,
        infer_timestamps_key=infer_timestamps_key,
        multi_value_prefix=multi_value_prefix,
        include_loss_weights=include_loss_weights,
        include_candidate_items=include_candidate_items,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collect_fn,
        shuffle=False,
        **_base_loader_kwargs(num_workers, prefetch_factor, pin_memory=True, persistent_workers=True)
    )

    return data_loader
