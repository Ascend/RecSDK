import os
import time
import torch
import argparse
import torch_npu
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from SIM import SIMModel, SIMLoss

torch_npu.npu.matmul.allow_hf32=True

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
args = parser.parse_args()

def generate_training_data(num_samples=1000, seq_len=1000, embedding_dim=64, num_categories=100):
    """生成训练数据"""
    print("生成训练数据...")
    user_behavior_sequences = torch.randn(num_samples, seq_len, embedding_dim)
    target_item_embeddings = torch.randn(num_samples, embedding_dim)
    behavior_categories = torch.randint(0, num_categories, (num_samples, seq_len))
    time_intervals = torch.randint(0, 100, (num_samples, 200))
    user_features = torch.randn(num_samples, 32)
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
            print(f'训练进度: {epoch}轮 [{batch_idx * len(user_behaviors)}/{len(train_loader.dataset)} '
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
            
            #compile
            if os.environ.get('MODEL_COMPILE_FLAG', "0") == "1":
                model = torch.compile(model, dynamic=False, backend="inductor")
                print("Inductor MODE YES")

            #profiling
            if os.environ.get('MODEL_PROFILING_FLAG', "0") == "1":
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
                steps = 10
                torch_npu.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                        ],
                    schedule=torch_npu.profiler.schedule(wait=6, warmup=0, active=1, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_modules=False,
                    with_flops=False,
                    experimental_config=experimental_config) as prof:
                    print("profiling start...")
                    for step in range(steps):
                        print(f"profiling steps == {step}")
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
                        torch_npu.npu.synchronize()
                        prof.step()
                        total_loss += loss.item()

            #E2E耗时
            if os.environ.get('MODEL_E2E_FLAG', "0") == "1":
                total_time = 0.0
                count = 0
                with torch.no_grad():
                    for i in range(100):
                        start_time = time.time()
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
                        end_time = time.time()
                        times = end_time - start_time
                        if i >= 50:
                            total_time += times
                            count += 1
                        total_loss += loss.item()
                avg_time = total_time / count * 1000
                print(f"E2E时间 = {avg_time:.4f}")

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
    device = torch.device('npu' if torch.npu.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成训练数据
    data = generate_training_data(num_samples=2000, seq_len=1000, embedding_dim=64, num_categories=100)
    
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
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型实例
    model = SIMModel(
        item_embedding_dim=64,
        user_feature_dim=32,
        hidden_dim=128,
        num_heads=8,
        dropout=0.1,
        search_type='hard',
        num_categories=100
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = SIMLoss(search_type='hard')
    
    # 训练参数
    num_epochs = 1
    
    print("开始训练...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch)
        print(f'第{epoch}轮训练完成，损失: {train_loss:.4f}')

    # 评估模型
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f'测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()
