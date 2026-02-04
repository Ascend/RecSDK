import os
import time
import torch
import argparse
import logging
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from SIM import SIMModel, SIMLoss


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32=True
elif torch.npu.is_available():
    import torch_npu
    torch_npu.npu.matmul.allow_hf32=True


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
args = parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
        torch_npu.npu.manual_seed_all(seed)

def save_tensor(pred, tensor_name = "pred_gpu.pt"):
    if isinstance(pred, torch.Tensor):
        tensor_to_save = pred.detach().cpu()
        logger.info(f"save tensor type: {type(tensor_to_save)}")
        torch.save(tensor_to_save, tensor_name)


def load_tensor(path):
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"{str(path)}：成功加载 tensor")
        return tensor
    except Exception as e:
        logger.info(f"{str(path)}：加载失败 -> {e}")
        return None


def compare_accuracy_diff(npu_output, gpu_output, verbose=True):
    results = {}

    # 平均绝对误差
    abs_error = torch.abs(npu_output - gpu_output)
    results['mean_abs_error'] = abs_error.mean().item()

    # 平均相对误差
    eps = 1e-8
    rel_error = abs_error / (torch.abs(npu_output) + eps)
    results['mean_rel_error'] = rel_error.mean().item()
    
    # 均方误差
    mse = torch.mean((npu_output - gpu_output) ** 2)
    rmse = torch.sqrt(mse)
    results['rmse'] = rmse.item()

    if verbose:
        logger.info(f"平均绝对误差: {results['mean_abs_error']:.6e}")
        logger.info(f"平均相对误差: {results['mean_rel_error']:.6e}")
        logger.info(f"均方根误差(RMSE): {results['rmse']:.6e}")
    return results


def compare_result(
        tensor1,
        tensor2,
        abs_error_threshold: float = 1e-4,
        rel_error_threshold: float = 1e-4,
):
    logger.info(f"loading tensor1 {tensor1} \nloading tensor2 {tensor2}")
    tensor_1 = load_tensor(tensor1)
    tensor_2 = load_tensor(tensor2)

    if tensor_1 is not None and tensor_2 is not None:
        try:
            torch.testing.assert_close(
                tensor_1, tensor_2,
                atol = abs_error_threshold,
                rtol = rel_error_threshold,
                equal_nan = True
            )
            logger.info("精度结果对齐")
            logger.info(compare_accuracy_diff(tensor_1, tensor_2))
        except AssertionError as e:
            logger.info("精度结果未对齐")
            logger.info(compare_accuracy_diff(tensor_1, tensor_2))


def save_mode_tensor(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for user_behaviors, target_items, behavior_cats, time_ints, user_feats, targets in test_loader:
            user_behaviors = user_behaviors.to(device)
            target_items = target_items.to(device)
            behavior_cats = behavior_cats.to(device)
            time_ints = time_ints.to(device)
            user_feats = user_feats.to(device)
            targets = targets.to(device)

            # save tensor
            if os.environ.get('SAVE_TENSOR_FLAG', "False").upper() == "TRUE":
                # save eager tensor
                ctr_predictions, _, _ = model(
                        user_behavior_seq=user_behaviors,
                        target_item_emb=target_items,
                        user_features=user_feats,
                        behavior_categories=behavior_cats,
                        time_intervals=time_ints
                    )
                logger.info("save eager tensor")

                if torch.cuda.is_available():
                    save_tensor(ctr_predictions, tensor_name = "pred_gpu.pt")
                elif torch.npu.is_available():
                    save_tensor(ctr_predictions, tensor_name = "pred_npu.pt")

                # save inductor tensor
                if os.environ.get('MODEL_COMPILE_FLAG', "False").upper() == "TRUE":
                    model = torch.compile(model, dynamic=False, backend="inductor")
                    ctr_predictions, _, _ = model(
                        user_behavior_seq=user_behaviors,
                        target_item_emb=target_items,
                        user_features=user_feats,
                        behavior_categories=behavior_cats,
                        time_intervals=time_ints
                    )
                    logger.info("save inductor tensor")

                    if torch.cuda.is_available():
                        save_tensor(ctr_predictions, tensor_name = "pred_gpu_inductor.pt")
                    elif torch.npu.is_available():
                        save_tensor(ctr_predictions, tensor_name = "pred_npu_inductor.pt")


def generate_training_data(num_samples=2000, seq_len=1000, item_embedding_dim=512, user_feature_dim=512, num_categories=100):
    """生成训练数据"""
    logger.info("生成训练数据...")
    user_behavior_sequences = torch.randn(num_samples, seq_len, item_embedding_dim)
    target_item_embeddings = torch.randn(num_samples, item_embedding_dim)
    behavior_categories = torch.randint(0, num_categories, (num_samples, seq_len))
    time_intervals = torch.randint(0, 100, (num_samples, 200))
    user_features = torch.randn(num_samples, user_feature_dim)
    target_ctrs = torch.randint(0, 2, (num_samples, 1)).float()
    
    return {
        'user_behavior_sequences': user_behavior_sequences,
        'target_item_embeddings': target_item_embeddings,
        'behavior_categories': behavior_categories,
        'time_intervals': time_intervals,
        'user_features': user_features,
        'target_ctrs': target_ctrs
    }


def train_model(model, train_loader, optimizer, criterion, device, epoch):
    """训练模型"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (user_behaviors, target_items, behavior_cats, time_ints, user_feats, targets) in enumerate(train_loader):
        user_behaviors = user_behaviors.to(device)
        target_items = target_items.to(device)
        behavior_cats = behavior_cats.to(device)
        time_ints = time_ints.to(device)
        user_feats = user_feats.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        ctr_predictions, _, _ = model(
            user_behavior_seq=user_behaviors,
            target_item_emb=target_items,
            user_features=user_feats,
            behavior_categories=behavior_cats,
            time_intervals=time_ints
        )
        
        # 计算损失
        loss = criterion(ctr_predictions, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 每隔 10 个 batch 输出一次训练进度
        if batch_idx % 10 == 0:
            logger.info(f'训练进度: {epoch}轮 [{batch_idx * len(user_behaviors)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches

    return avg_loss


def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for user_behaviors, target_items, behavior_cats, time_ints, user_feats, targets in test_loader:
            user_behaviors = user_behaviors.to(device)
            target_items = target_items.to(device)
            behavior_cats = behavior_cats.to(device)
            time_ints = time_ints.to(device)
            user_feats = user_feats.to(device)
            targets = targets.to(device)
            
            # compile mode
            if os.environ.get('MODEL_COMPILE_FLAG', "False").upper() == "TRUE":
                model = torch.compile(model, dynamic=False, backend="inductor")
                logger.info("Inductor MODE YES")
            elif os.environ.get('MODEL_ACLGRAPH_FLAG', "False").upper() == "TRUE":
                model = torch.compile(model, dynamic=False, backend="inductor", mode="reduce-overhead")
                logger.info("Aclgraph MODE YES")

            # 采集profiling
            if os.environ.get('MODEL_PROFILING_FLAG', "False").upper() == "TRUE":
                # 运行step次数
                steps = 30
                # 空跑预热
                for step in range(steps):
                    # 前向传播
                    ctr_predictions, _, _ = model(
                        user_behavior_seq=user_behaviors,
                        target_item_emb=target_items,
                        user_features=user_feats,
                        behavior_categories=behavior_cats,
                        time_intervals=time_ints
                    )
                    # 计算损失
                    loss = criterion(ctr_predictions, targets)
                    total_loss += loss.item()

                # GPU profiling
                if torch.cuda.is_available():
                    # 正式采集profiling数据
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA
                            ],
                        # 等待24步，采集第25步，重复1次
                        schedule=torch.profiler.schedule(
                            wait=24, 
                            warmup=0, 
                            active=1, 
                            repeat=1
                        ),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler("./GPU_profiling_result"),
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=True
                        ) as prof:
                        logger.info("profiling start...")
                        for step in range(steps):
                            logger.info(f"profiling steps == {step}")
                            # 前向传播
                            ctr_predictions, _, _ = model(
                                user_behavior_seq=user_behaviors,
                                target_item_emb=target_items,
                                user_features=user_feats,
                                behavior_categories=behavior_cats,
                                time_intervals=time_ints
                            )
                            # 计算损失
                            loss = criterion(ctr_predictions, targets)
                            total_loss += loss.item()
                            prof.step()

                # NPU profiling
                elif torch.npu.is_available():
                    # 正式采集profiling数据
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
                        export_type=[
                            torch_npu.profiler.ExportType.Text,
                            torch_npu.profiler.ExportType.Db
                            ],
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None
                        )
                    with torch_npu.profiler.profile(
                        activities=[
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU
                            ],
                        # 等待24步，采集第25步，重复1次
                        schedule=torch_npu.profiler.schedule(
                            wait=24, 
                            warmup=0, 
                            active=1, 
                            repeat=1
                        ),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./NPU_profiling_result"),
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=True,
                        with_modules=False,
                        with_flops=False,
                        experimental_config=experimental_config) as prof:
                        logger.info("profiling start...")
                        for step in range(steps):
                            logger.info(f"profiling steps == {step}")
                            # 前向传播
                            ctr_predictions, _, _ = model(
                                user_behavior_seq=user_behaviors,
                                target_item_emb=target_items,
                                user_features=user_feats,
                                behavior_categories=behavior_cats,
                                time_intervals=time_ints
                            )
                            # 计算损失
                            loss = criterion(ctr_predictions, targets)
                            total_loss += loss.item()
                            prof.step()

            #统计E2E耗时
            if os.environ.get('MODEL_E2E_FLAG', "False").upper() == "TRUE":
                latency_list = []
                warmup_steps = 50
                total_steps = 100
                with torch.no_grad():
                    # 空跑预热
                    for i in range(warmup_steps):
                        ctr_predictions, _, _ = model(
                            user_behavior_seq=user_behaviors,
                            target_item_emb=target_items,
                            user_features=user_feats,
                            behavior_categories=behavior_cats,
                            time_intervals=time_ints
                        )
                        loss = criterion(ctr_predictions, targets)
                        total_loss += loss.item()
                    # 正式统计E2E
                    start_time = time.time()
                    for i in range(total_steps):
                        # 前向传播
                        step_start_time = time.time()
                        ctr_predictions, _, _ = model(
                            user_behavior_seq=user_behaviors,
                            target_item_emb=target_items,
                            user_features=user_feats,
                            behavior_categories=behavior_cats,
                            time_intervals=time_ints
                        )
                        # 计算损失
                        loss = criterion(ctr_predictions, targets)
                        total_loss += loss.item()
                        step_end_time = time.time()
                        latency = (step_end_time - step_start_time) * 1000
                        latency_list.append(latency)
                    end_time = time.time()
                    e2e_time = (end_time - start_time) * 1000
                e2e_avg_time = round(e2e_time / total_steps, 6)
                step_avg_time = round(sum(latency_list) / total_steps, 6)

                latency_list.sort()
                p90 = 0.0
                p99 = 0.0

                if len(latency_list) > 0:
                    p90_index = int(len(latency_list) * 0.9)
                    p90_index = min(p90_index, len(latency_list) - 1)
                    p90 = round(latency_list[p90_index], 6)

                    p95_index = int(len(latency_list) * 0.95)
                    p95_index = min(p95_index, len(latency_list) - 1)
                    p95 = round(latency_list[p95_index], 6)

                    p99_index = int(len(latency_list) * 0.99)
                    p99_index = min(p99_index, len(latency_list) - 1)
                    p99 = round(latency_list[p99_index], 6)

                    p999_index = int(len(latency_list) * 0.999)
                    p999_index = min(p999_index, len(latency_list) - 1)
                    p999 = round(latency_list[p999_index], 6)

                QPS = int(args.batch_size / e2e_avg_time * 1000)
                
                E2E_RESULT = {
                    "Batch_size": args.batch_size,
                    "model_name": "SIM",
                    "QPS": QPS,
                    "AVG Latency": e2e_avg_time,
                    "P999 Latency": p999,
                    "P99 Latency": p99,
                    "P95 Latency": p95,
                    "P90 Latency": p90
                }
                logger.info(E2E_RESULT)

            # 前向传播
            ctr_predictions, _, _ = model(
                user_behavior_seq=user_behaviors,
                target_item_emb=target_items,
                user_features=user_feats,
                behavior_categories=behavior_cats,
                time_intervals=time_ints
            )
            
            # 计算损失
            loss = criterion(ctr_predictions, targets)
            total_loss += loss.item()
            
            # 计算准确率
            predicted_labels = (ctr_predictions > 0.5).float()
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def main():
    """主训练函数"""
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.npu.is_available():
        device = torch.device('npu')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置超参并且生成训练数据
    item_embedding_dim=512
    user_feature_dim=512
    hidden_dim=1024
    data = generate_training_data(num_samples=2000, seq_len=1000, item_embedding_dim=item_embedding_dim, user_feature_dim=user_feature_dim, num_categories=100)
    
    # 创建数据集
    dataset = TensorDataset(
        data['user_behavior_sequences'],
        data['target_item_embeddings'],
        data['behavior_categories'],
        data['time_intervals'],
        data['user_features'],
        data['target_ctrs']
    )
    
    # 分割训练和测试数据
    train_size = int(len(dataset) - args.batch_size)
    test_size = args.batch_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型实例
    model = SIMModel(
        item_embedding_dim=item_embedding_dim,
        user_feature_dim=user_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        search_type='hard',
        num_categories=100
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = SIMLoss(search_type='hard')
    
    # 训练模型
    if os.environ.get('MODEL_MODE', "INFER").upper() == "TRAIN":
        num_epochs = 1
        logger.info("开始训练...")
        for epoch in range(1, num_epochs + 1):
            train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch)
            logger.info(f'第{epoch}轮训练完成，损失: {train_loss:.4f}')
            # 评估模型
            test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
            logger.info(f'测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}')

    # 模型推理
    elif os.environ.get('MODEL_MODE', "INFER").upper() == "INFER":
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        logger.info(f'测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}')

    # 保存模型tensor
    if os.environ.get('SAVE_TENSOR_FLAG', "False").upper() == "TRUE":
        save_mode_tensor(model, test_loader, device)

    # 模型精度比较
    if os.environ.get('COMPARE_ACCURACY_FLAG', "False").upper() == "TRUE":
        if os.environ.get('MODEL_COMPILE_FLAG', "False").upper() == "TRUE":
            if torch.cuda.is_available():
                compare_result(tensor1= "pred_gpu.pt",tensor2 = "pred_gpu_inductor.pt")
            elif torch.npu.is_available():
                compare_result(tensor1= "pred_npu.pt",tensor2 = "pred_npu_inductor.pt")
        else:
            compare_result(tensor1= "pred_npu.pt",tensor2 = "pred_gpu.pt")

if __name__ == "__main__":
    # 固定随机数种子
    set_seed(2025)
    main()
