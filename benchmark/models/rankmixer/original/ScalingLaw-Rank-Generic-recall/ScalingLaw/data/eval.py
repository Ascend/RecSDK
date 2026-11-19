import torch
import torch.distributed as dist

from modeling.generic.sequential_v2.features import SequentialFeatures

import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    precision_recall_curve,
    auc
)

from utils.common_utils import weird_division
from utils.common_utils import Const

class MetricsCalculator:
    def __init__(self, eval_df, featuremap_dict=None, metric_list=None, group_metric_cols=None,
                 bias_evaluation_conf=None, save_result_to_local=True):
        self.eval_df = eval_df.copy()
        self.y_true = self.eval_df['ground_truth']
        self.y_score = self.eval_df['scores']

        self.metric_list = metric_list
        self.group_metric_cols = group_metric_cols
        self.bias_evaluation_conf = bias_evaluation_conf
        self.save_result_to_local = save_result_to_local

        self.featuremap_dict = featuremap_dict
        logging.info(f'print eval_df keys is {eval_df.keys()}')
        # 注册指标计算方法
        self.metric_funcs = {
            'auc': self._calc_auc,
            'logloss': self._calc_logloss,
            'bucket_copc': self._calc_bucket_copc,
            'pcoc': self._calc_pcoc,
            'count': self._calc_count,
            'mean': self._calc_mean,
            'variance': self._calc_variance,
            'stddev': self._calc_stddev,
            'segments_copc': self._calc_segments_copc,
            'prauc': self._calc_prauc,
            'ece': self._calc_ece,
            'gauc': self._calc_gauc,
            'gpcoc': self._calc_gpcoc,
            'caln': self._calc_caln,
            'gece': self._calc_gece,
        }

    def calculate(self, output_path):
        # 计算全局指标
        global_results = self._calculate_metrics(self.metric_list, dataset_name="global")

        # 计算分组指标
        group_results = {}
        for group in self.group_metric_cols:
            # 生成分组查询条件
            query_conditions = []
            group_name_parts = []
            
            for col, values in group.items():
                if isinstance(values, list) and col in self.eval_df.columns:
                    if all(isinstance(v, int) for v in values):
                        _values = values
                    else:
                        _values = []
                        for v in values:
                            if col not in self.featuremap_dict:
                                logging.warning(f'column name {col} not found in values, trying query {v} instead')
                                result = self.featuremap_dict.get(v, None)
                            else:
                                result = self.featuremap_dict[col].get(v, None)
                            if result is not None:
                                _values.append(result)
                            else:
                                logging.warning(f'column name {v} not found in values')
                    if len(_values) > 0:
                        condition = f"{col} in {_values}" if len(_values) > 1 else f"{col} == {_values[0]}"
                        query_conditions.append(condition)
                        group_name_parts.append(f"{col}={'_'.join(map(str, _values))}")

            if query_conditions:
                # 生成分组名称
                group_name = "_AND_".join(group_name_parts)

                # 筛选数据子集
                combined_query = " & ".join(query_conditions)
                logging.info(f'printing query conditions {query_conditions}')
                logging.info(f'printing combined query {combined_query}')
                subgroup_df = self.eval_df.query(combined_query, engine='python')

                # 子集计算指标
                if not subgroup_df.empty:
                    subgroup_calculator = MetricsCalculator(subgroup_df, None, )
                    group_results[group_name] = subgroup_calculator._calculate_metrics(self.metric_list, group_name)
                else:
                    group_results[group_name] = {k: None for k in global_results.keys()}

        # 计算偏差评估
        bias_results = {}
        for conf in self.bias_evaluation_conf:
            key = f"{conf['high_level_dimension']}_{conf['low_level_dimension']}"
            bias_results[key] = self._calculate_bias_metrics(
                self.eval_df,
                conf.get('mode', 'all'),
                conf['high_level_dimension'],
                conf['low_level_dimension']
            )

        # 合并结果
        results = {
            "bias_evaluation": bias_results,
            "groups": group_results,
            "global": global_results
        }

        self.print_results(results)
        if self.save_result_to_local:
            self._export_to_csv(results, output_path)

    def _calculate_metrics(self, metric_list, dataset_name=""):
        """内部计算方法"""
        results = {}
        for metric in metric_list:
            tag = metric['tag']
            start_time = time.time()
            func = self.metric_funcs.get(tag)
            if not func:
                logging.warning("[%s-%s] skip metric not supported", dataset_name, tag)
                continue

            config = metric.get('config',{})
            try:
                value = func(**config)
                results[metric.get('name', tag)] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("[%s-%s] skip metric: %s", dataset_name, tag, str(e))
                results[metric.get('name', tag)] = None
            except Exception as e:
                logging.warning("[%s-%s] metric calc failed: %s", dataset_name, tag, str(e), exc_info=True)
                results[metric.get('name', tag)] = None
            end_time = time.time()
            logging.info('finished calculation of metrics %s of %s, cost: %.2f s, value: %s', tag, dataset_name,
                         end_time - start_time, results.get(metric.get('name', tag)))
        return results
    
    def _calculate_bias_metrics(self, df, mode, high_dim, low_dim):
        """引入误差信息和偏差大小进行评估"""
        metrics = {}

        start_time = time.time()

        if mode not in ["all", "mvce", "gc_n"]:
            if not mode:
                mode = 'all'
            else:
                logging.warning("skip bias_metric not supported: %s", mode)
                return metrics

        # 多维度校准误差
        if mode in ["mvce", "all"]:
            tag = "MVCE"
            try:
                groups = df.groupby(high_dim, observed=False)
                value = groups.apply(self._calculate_pce, bin_col=low_dim).mean()
                metrics[tag] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("[MVCE] skip metric: %s", str(e))
                metrics[tag] = None
            except Exception as e:
                logging.warning("[MVCE] metric calc failed: %s", str(e), exc_info=True)
                metrics[tag] = None
            logging.info('finished calculation of metrics MVCE, value: %s', metrics.get(tag))

        # 分组校准指标
        if mode in ["gc_n", "all"]:
            tag = "GC-N"
            try:
                grouped = df.groupby([high_dim, low_dim], observed=False)
                error_df = grouped.apply(self._calculate_error).reset_index(name='error_i')
                value = error_df.groupby(high_dim, observed=False).apply(self._calculate_gc_n).mean()
                metrics[tag] = round(value, 5) if isinstance(value, float) else value
            except ValueError as e:
                logging.warning("[GC-N] skip metric: %s", str(e))
                metrics[tag] = None
            except Exception as e:
                logging.warning("[GC-N] metric calc failed: %s", str(e), exc_info=True)
                metrics[tag] = None
            logging.info('finished calculation of metrics GC-N, value: %s', metrics.get(tag))
        end_time = time.time()
        logging.info('finished calculation of bias metrics %s - %s, cost: %.2f s', high_dim, low_dim,
                     end_time - start_time)
        return metrics

    def _export_to_csv(self, results, output_path):
        """
        将指标结果导出为CSV文件
        
        参数:
            results: calculate()方法返回的结果字典
            output_path: 输出文件路径，如 'metrics_report.csv'
        """
        try:
            records = []

            # 处理全局指标
            global_metrics = results['global']
            records.append({
                'scope': 'global',
                'group_name': 'global',
                **global_metrics
            })

            # 处理分组指标
            for group_name, metrics in results['groups'].items():
                records.append({
                    'scope': 'group',
                    'group_name': group_name,
                    **metrics
                })

            df = pd.DataFrame(records)

            # 调整列顺序
            columns = ['scope', 'group_name'] + [col for col in df.columns if col not in ('scope', 'group_name')]
            df = df[columns]

            df.to_csv(output_path, index=False, float_format='%.5f')
            logging.info("CSV exported, path: %s, count: %d", output_path, len(records))
        except PermissionError:
            logging.error("File write permission denied, path: %s", output_path, exc_info=True)
            raise
        except Exception as e:
            logging.error("Unknown error during CSV export: %s", str(e), exc_info=True)
            raise

    def print_results(self, results, indent=0):
        prefix = " " * indent
        for k,v in results.items():
            if isinstance(v,dict):
                print(f"{prefix}{k}:")
                self.print_results(v, indent + 4)
            else:
                print(f"{prefix}{k}: {v}")
    
    def _calc_auc(self):
        return roc_auc_score(self.y_true, self.y_score)
    
    def _calc_logloss(self):
        return log_loss(self.y_true, self.y_score)
    
    def _calc_bucket_copc(self, buckets=10):
        """分桶COPC"""
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        eval_df['bucket'] = pd.qcut(eval_df['scores'], buckets, duplicates='drop')
        grouped = eval_df.groupby('bucket', observed=False).agg(
            pctr=('scores', 'mean'),
            ctr=('ground_truth', 'mean')
        )
        grouped = grouped[grouped.ctr > 0]
        return (grouped.pctr / grouped.ctr).mean()

    def _calc_pcoc(self):
        """预测点击率与实际点击率比值"""
        pctr = self.y_score.mean()
        ctr = self.y_true.mean()
        return pctr / ctr if ctr != 0 else 0.0

    def _calc_count(self):
        return len(self.y_score)

    def _calc_mean(self):
        return self.y_score.mean()

    def _calc_variance(self):
        return self.y_score.var()

    def _calc_stddev(self):
        return self.y_score.std()

    def _calc_segments_copc(self, copc_pctr_segments):
        """指定分桶点COPC"""
        bins = list(map(float, copc_pctr_segments.split('#')))
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        eval_df['bucket'] = pd.cut(eval_df['scores'], bins=bins, include_lowest=True)
        grouped = eval_df.groupby('bucket', observed=False).agg(
            sum_pred=('scores', 'sum'),
            sum_true=('ground_truth', 'sum'),
            count=('scores', 'count')
        )
        grouped['copc'] = np.divide(
            grouped['sum_pred'] + Const.EPS,
            grouped['sum_true'] + Const.EPS,
            out=np.zeros_like(grouped['sum_pred']),  # 分母为零时输出0
            where=(grouped['sum_true']) != 0
        )
        return grouped['copc'].tolist()

    def _calc_prauc(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        return auc(recall, precision)

    def _calc_ece(self, n_bins=10):
        eval_df = self.eval_df.loc[:, ['scores', 'ground_truth']]
        bins = np.linspace(0, 1, n_bins + 1)
        labels = range(n_bins)
        eval_df["bin"] = pd.cut(
            self.y_score, bins=bins, labels=labels, include_lowest=True
        )
        return self._calculate_pce(eval_df, "bin")

    def _gen_group_samples(self, gid, group_sample_rate=1.0):
        group_df = self.eval_df.loc[:, [gid, 'ground_truth', 'scores']]
        group_df[gid] = group_df[gid].astype('category')
        if group_sample_rate < 1.0 and group_sample_rate > 0.0:
            sampled_gids = group_df[gid].drop_duplicates().sample(frac=group_sample_rate, random_state=42)
            group_df = group_df[group_df[gid].isin(sampled_gids)]
        return group_df.groupby(gid, observed=False)
    
    def _calc_gauc(self, gid, group_sample_rate=1.0):
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        total_auc = 0.0
        total_weight = 0
        for _, group in grouped:
            y_true = group['ground_truth']
            if y_true.nunique() < 2:
                continue
            total_auc += roc_auc_score(y_true, group['scores']) * len(y_true)
            total_weight += len(y_true)
        return total_auc / total_weight if total_weight > 0 else 0.0

    def _calc_gpcoc(self, gid, group_sample_rate=1.0):
        """分组PCOC"""
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        grouped = grouped.agg(
            pctr=('scores', 'mean'),
            ctr=('ground_truth', 'mean')
        )
        # grouped.ctr为0的数据代表无正样本，在此基础上无法计算pcoc，因此需要过滤
        grouped = grouped[grouped.ctr > 0]
        return (grouped.pctr / grouped.ctr).mean()

    def _calc_caln(self, gid, group_sample_rate=1.0):
        grouped = self._gen_group_samples(gid, group_sample_rate=1.0)
        return np.sqrt(grouped.apply(self._calculate_error).mean(skipna=True))

    def _calc_gece(self, gid, group_sample_rate=1.0):
        group_df = self.eval_df.loc[:, [gid, 'ground_truth', 'scores']]
        group_df[gid] = group_df[gid].astype('category')
        if group_sample_rate < 1.0 and group_sample_rate > 0.0:
            sampled_gids = group_df[gid].drop_duplicates().sample(frac=group_sample_rate, random_state=42)
            group_df = group_df[group_df[gid].isin(sampled_gids)]
        return self._calculate_pce(group_df, gid)

    def _calculate_pce(self, group, bin_col='bin'):
        pce = 0.0
        total_samples = len(group)

        for _, bin_group in group.groupby(bin_col, observed=False):
            n_b = len(bin_group)
            if n_b == 0:
                continue
            conf_b = bin_group["scores"].mean()
            acc_b = bin_group["ground_truth"].mean()
            pce += (n_b / total_samples) * abs(acc_b - conf_b)
        return pce

    def _calculate_error(self, group):
        sum_gt = group['ground_truth'].sum()
        if sum_gt == 0:
            return np.nan
        sum_scores = group['scores'].sum()
        ratio = weird_division(sum_gt, sum_scores) if sum_gt > sum_scores else weird_division(sum_scores, sum_gt)
        return (ratio - 1) ** 2

    def _calculate_gc_n(self, group):
        mean_squared = group['error_i'].mean()
        return np.sqrt(mean_squared)
    


@torch.inference_mode
def process_rank_scores(model, seq_features: SequentialFeatures, save_score_csv_item_cols,
                                      save_score_csv_user_cols, token_per_item=2):
    """
    使用模型和序列特征评估排名指标。

    :param model: 用于评估的模型。
    :param seq_features: 用于 eval 的历史序列特征。
    :param test_positions: 测试位置的张量。
    :return: 分数张量、真实标签和group_key。
    """
    model_input = seq_features.past_payloads
    model_input['past_lengths'] = seq_features.past_lengths
    model_input['past_ids'] = seq_features.past_ids
    model_input['uid'] = seq_features.uid
    scores = scores["rerank_score"]
    supervision_ids = seq_features.past_payloads["candidate_item_id"]
    ground_truth = seq_features.past_payloads["candidate_labels"]
    # 收集所有需要放进结果 DataFrame 的字段
    eval_group_keys = {
        'uid': seq_features.uid.unsqueeze(-1) * torch.ones_like(supervision_ids)
    }
    for key in save_score_csv_item_cols:
        if "candidate_" + key not in seq_features.past_payloads.keys():
            raise ValueError("candidate_" + key + " not in past_payloads")
        eval_group_keys[key] = seq_features.past_payloads["candidate_" + key]
    for key in save_score_csv_user_cols:
        if key not in seq_features.past_payloads.keys():
            raise ValueError(key + " not in past_payloads")
        eval_group_keys[key] = seq_features.past_payloads[key].unsqueeze(-1) * torch.ones_like(supervision_ids)

    # 由于之前对candidate有复制操作，这里切片去重
    eval_scores = scores[supervision_ids != 0][::token_per_item]
    eval_ground_truth = ground_truth[supervision_ids != 0][::token_per_item]
    for key in eval_group_keys.keys():
        eval_group_keys[key] = eval_group_keys[key][supervision_ids != 0][::token_per_item]
    
    return eval_scores, eval_ground_truth, eval_group_keys

def avg_eval(x: torch.Tensor, world_size: int) -> float:
    """
    计算分布式环境中的平均评估值。

    :param x: 输入的张量。
    :param world_size: 分布式环境中的进程数。
    :return: 平均值。
    """
    _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32, device=x.device)
    if world_size > 1:
        dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
    return torch.div(_sum_and_numel[0], _sum_and_numel[1])

def gather_all_list(x: list, world_size: int) -> torch.Tensor:
    obj_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj_list, x, group=None)
    l = []
    for o in obj_list:
        if isinstance(o, list):
            l.extend(o)
        else:
            l.extend(o.tolist())
    return l
