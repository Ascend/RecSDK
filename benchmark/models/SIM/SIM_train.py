import os
import gc
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
parser.add_argument(
    "--enable_dynamic_compile",
    type=lambda v: {'true': True, 'false': False, 'none': None}[v.lower()],
    default=False,
    help="Dynamic compile mode: False (static), True (full dynamic), None (auto detect)",
)
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

def save_tensor(pred, device_name, model_name, inductor_flag):
    if isinstance(pred, torch.Tensor):
        tensor_to_save = pred.detach().cpu()
        logger.info(f"save tensor type: {type(tensor_to_save)}")
        os.makedirs(f"../save_results_{device_name}/{model_name}", exist_ok=True)
        torch.save(tensor_to_save, f"../save_results_{device_name}/{model_name}/predictions_{model_name}_{inductor_flag}_0.pt")

def parse_shape_list():
    """解析SHAPE_LIST环境变量"""
    shape_list = []
    shape_list_str = os.getenv("SHAPE_LIST", "")
    if shape_list_str:
        for pair in shape_list_str.split(";"):
            pair = pair.strip()
            if pair:
                parts = pair.split(",")
                bs = int(parts[0].strip())
                sl = int(parts[1].strip()) if len(parts) > 1 else 1000
                shape_list.append((bs, sl))
    return shape_list

def generate_sim_batch(bs, seq_len, device, item_embedding_dim=512, user_feature_dim=512, num_categories=100):
    """生成模型的输入数据"""
    user_behavior_seq = torch.randn(bs, seq_len, item_embedding_dim, device=device)
    target_item_emb = torch.randn(bs, item_embedding_dim, device=device)
    behavior_categories = torch.randint(0, num_categories, (bs, seq_len), device=device)
    # time_intervals 固定 100 维即可（模型内部会按 k=min(100, seq_len) 截断/补齐）
    time_intervals = torch.randint(0, 100, (bs, 100), device=device)
    user_features = torch.randn(bs, user_feature_dim, device=device)
    target_ctrs = torch.randint(0, 2, (bs, 1), device=device).float()
    return user_behavior_seq, target_item_emb, behavior_categories, time_intervals, user_features, target_ctrs


def mark_sim_dynamic(user_behavior_seq, target_item_emb, behavior_categories, time_intervals, user_features):
    """标记动态维度"""
    torch._dynamo.mark_dynamic(user_behavior_seq, 0)
    torch._dynamo.mark_dynamic(user_behavior_seq, 1)
    torch._dynamo.mark_dynamic(behavior_categories, 0)
    torch._dynamo.mark_dynamic(behavior_categories, 1)
    torch._dynamo.mark_dynamic(target_item_emb, 0)
    torch._dynamo.mark_dynamic(user_features, 0)
    torch._dynamo.mark_dynamic(time_intervals, 0)

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
                if os.environ.get('MODEL_ACLGRAPH_FLAG', "False").upper() == "TRUE":
                    model = torch.compile(model, dynamic=False, backend="inductor", mode="reduce-overhead")
                    logger.info("Inductor Aclgraph MODE YES")
                else:
                    model = torch.compile(model, dynamic=False, backend="inductor")
                    logger.info("Inductor MODE YES")

            inductor_flag = 'inductor' if os.environ.get('MODEL_COMPILE_FLAG', 'False').upper() == 'TRUE' else 'eager'
            model_name = os.environ.get("MODEL_NAME", "default_name")
            device_name = device.type
            model_detail_info = device_name + "_" + model_name + "_" + inductor_flag

            #统计E2E耗时
            if os.environ.get('MODEL_E2E_FLAG', "False").upper() == "TRUE":
                gc.disable()
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
                p999 = 0.0

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

                eval_str = "performance: " + model_detail_info + "_" + str(E2E_RESULT)
                os.makedirs(f"../save_results_{device_name}/", exist_ok=True)
                with open(f"../save_results_{device_name}/performance_result.txt", "a", encoding="utf-8") as f:
                    f.write(eval_str + "\n")
                logger.info(eval_str)

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
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join("../profiling", model_name)),
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
                            torch_npu.profiler.ExportType.Text
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
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            os.path.join("../profiling", model_name),
                            worker_name=model_detail_info,
                        ),
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

            # 前向传播
            ctr_predictions, _, _ = model(
                user_behavior_seq=user_behaviors,
                target_item_emb=target_items,
                user_features=user_feats,
                behavior_categories=behavior_cats,
                time_intervals=time_ints
            )
            
            # 保存模型tensor
            save_tensor(ctr_predictions, device_name, model_name, inductor_flag)
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

def evaluate_model_dynamic_shape(model, device, shape_list, criterion):
    """动态 shape 性能测试仅在 SHAPE_LIST 非空时调用，不影响原有的DataLoader推理流程。"""
    model.eval()

    with torch.no_grad():
        if os.environ.get('MODEL_COMPILE_FLAG', "False").upper() == "TRUE":
            dynamic_arg = args.enable_dynamic_compile
            if os.environ.get('MODEL_ACLGRAPH_FLAG', "False").upper() == "TRUE":
                model = torch.compile(model, dynamic=dynamic_arg, backend="inductor", mode="reduce-overhead")
                logger.info("Inductor Aclgraph MODE YES")
            else:
                model = torch.compile(model, dynamic=dynamic_arg, backend="inductor")
                logger.info("Inductor MODE YES")

        shape_idx = 0

        def next_shape():
            nonlocal shape_idx
            bs, sl = shape_list[shape_idx % len(shape_list)]
            shape_idx += 1
            return bs, sl

        def run_once(bs, sl):
            user_behaviors, target_items, behavior_cats, time_ints, user_feats, target_ctrs = generate_sim_batch(bs, sl, device)
            if args.enable_dynamic_compile is None:
                mark_sim_dynamic(user_behaviors, target_items, behavior_cats, time_ints, user_feats)
            ctr_predictions, _, _ = model(
                user_behavior_seq=user_behaviors,
                target_item_emb=target_items,
                user_features=user_feats,
                behavior_categories=behavior_cats,
                time_intervals=time_ints
            )
            return ctr_predictions, target_ctrs

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        iter_num = 210
        for it in range(iter_num):
            bs, sl = next_shape()
            logger.info(f"[iter {it}] shape=({bs}, {sl})")
            ctr_predictions, target_ctrs = run_once(bs, sl)

            loss = criterion(ctr_predictions, target_ctrs)
            total_loss += loss.item()
            predicted_labels = (ctr_predictions > 0.5).float()
            correct_predictions += (predicted_labels == target_ctrs).sum().item()
            total_samples += target_ctrs.size(0)

        inductor_flag = 'inductor' if os.environ.get('MODEL_COMPILE_FLAG', 'False').upper() == 'TRUE' else 'eager'
        model_name = os.environ.get("MODEL_NAME", "default_name")
        device_name = device.type
        model_detail_info = device_name + "_" + model_name + "_" + inductor_flag

        # 统计E2E耗时
        if os.environ.get('MODEL_E2E_FLAG', "False").upper() == "TRUE":
            gc.disable()
            latency_list = []
            e2e_batches_list = []
            warmup_steps = 50
            total_steps = 100
            # 空跑预热
            for i in range(warmup_steps):
                bs, sl = next_shape()
                _ = run_once(bs, sl)
            # 正式统计E2E
            start_time = time.time()
            for i in range(total_steps):
                bs, sl = next_shape()
                step_start_time = time.time()
                ctr_predictions, target_ctrs = run_once(bs, sl)
                _ = criterion(ctr_predictions, target_ctrs)
                step_end_time = time.time()
                latency = (step_end_time - step_start_time) * 1000
                latency_list.append(latency)
                e2e_batches_list.append(bs)
            end_time = time.time()
            e2e_time = (end_time - start_time) * 1000
            e2e_avg_time = round(e2e_time / total_steps, 6)

            latency_list.sort()
            p90 = 0.0
            p99 = 0.0
            p999 = 0.0

            if len(latency_list) > 0:
                p90_index = min(int(len(latency_list) * 0.9), len(latency_list) - 1)
                p90 = round(latency_list[p90_index], 6)

                p95_index = min(int(len(latency_list) * 0.95), len(latency_list) - 1)
                p95 = round(latency_list[p95_index], 6)

                p99_index = min(int(len(latency_list) * 0.99), len(latency_list) - 1)
                p99 = round(latency_list[p99_index], 6)

                p999_index = min(int(len(latency_list) * 0.999), len(latency_list) - 1)
                p999 = round(latency_list[p999_index], 6)

            QPS = int(sum(e2e_batches_list) / (e2e_time / 1000))

            E2E_RESULT = {
                "Batch_size": "Dynamic" if len(shape_list) > 1 else str(shape_list[0][0]),
                "model_name": "SIM",
                "QPS": QPS,
                "AVG Latency": e2e_avg_time,
                "P999 Latency": p999,
                "P99 Latency": p99,
                "P95 Latency": p95,
                "P90 Latency": p90
            }
            logger.info(E2E_RESULT)

            eval_str = "performance: " + model_detail_info + "_" + str(E2E_RESULT)
            os.makedirs(f"../save_results_{device_name}/", exist_ok=True)
            with open(f"../save_results_{device_name}/performance_result.txt", "a", encoding="utf-8") as f:
                f.write(eval_str + "\n")
            logger.info(eval_str)

        # 采集profiling
        if os.environ.get('MODEL_PROFILING_FLAG', "False").upper() == "TRUE":
            # 运行step次数
            steps = 30
            # 空跑预热
            for step in range(steps):
                bs, sl = next_shape()
                _ = run_once(bs, sl)

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
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join("../profiling", model_name)),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=True
                    ) as prof:
                    logger.info("profiling start...")
                    for step in range(steps):
                        logger.info(f"profiling steps == {step}")
                        bs, sl = next_shape()
                        ctr_predictions, target_ctrs = run_once(bs, sl)
                        _ = criterion(ctr_predictions, target_ctrs)
                        prof.step()

            # NPU profiling
            elif torch.npu.is_available():
                # 正式采集profiling数据
                experimental_config = torch_npu.profiler._ExperimentalConfig(
                    export_type=[
                        torch_npu.profiler.ExportType.Text
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
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        os.path.join("../profiling", model_name),
                        worker_name=model_detail_info,
                    ),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=True,
                    with_modules=False,
                    with_flops=False,
                    experimental_config=experimental_config) as prof:
                    logger.info("profiling start...")
                    for step in range(steps):
                        logger.info(f"profiling steps == {step}")
                        bs, sl = next_shape()
                        ctr_predictions, target_ctrs = run_once(bs, sl)
                        _ = criterion(ctr_predictions, target_ctrs)
                        prof.step()

        avg_loss = total_loss / iter_num
        accuracy = correct_predictions / total_samples
        logger.info(f'测试集损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')
    
    return avg_loss, accuracy


def create_model(item_embedding_dim, user_feature_dim, hidden_dim, device):
    return SIMModel(
        item_embedding_dim=item_embedding_dim,
        user_feature_dim=user_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        search_type='hard',
        num_categories=100
    ).to(device)

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

    # 动态shape模式
    shape_list = parse_shape_list()
    if shape_list and os.environ.get('MODEL_MODE', "INFER").upper() == "INFER":
        model = create_model(item_embedding_dim, user_feature_dim, hidden_dim, device)
        criterion = SIMLoss(search_type='hard')
        # 精度对比
        if os.environ.get('MODEL_COMPILE_PRECISION', '0') == '1':
            origin_model = model
            model = torch.compile(model, dynamic=args.enable_dynamic_compile, backend="inductor")
            set_seed(2026)
            test_loss, test_accuracy = evaluate_model_dynamic_shape(model, device, shape_list, criterion)
            set_seed(2026)
            test_loss_ori, test_accuracy_ori = evaluate_model_dynamic_shape(origin_model, device, shape_list, criterion)
            print(f"{test_loss=}")
            print(f"{test_loss_ori=}")
            print(f"{test_accuracy=}")
            print(f"{test_accuracy_ori=}")
            print("Inductor MODE YES")
            return
        evaluate_model_dynamic_shape(model, device, shape_list, criterion)
        return
    
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
    model = create_model(item_embedding_dim, user_feature_dim, hidden_dim, device)
    
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


if __name__ == "__main__":
    # 固定随机数种子
    set_seed(2025)
    main()
