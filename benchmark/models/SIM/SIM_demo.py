import torch
from SIM import SIMModel

def test_model_basic_functionality():
    """测试模型基本功能"""
    print("开始测试模型基本功能...")
    
    # 创建模型实例
    model = SIMModel(
        item_embedding_dim=64,
        user_feature_dim=32,
        hidden_dim=128,
        num_heads=8,
        dropout=0.1,
        search_type='hard',
        num_categories=100
    )
    print("模型创建成功")
    
    # 创建测试数据
    batch_size = 1
    seq_len = 100
    embedding_dim = 64
    
    user_behavior_sequences = torch.randn(batch_size, seq_len, embedding_dim)
    target_item_embeddings = torch.randn(batch_size, embedding_dim)
    behavior_categories = torch.randint(0, 100, (batch_size, seq_len))
    time_intervals = torch.randint(0, 100, (batch_size, 50))
    user_features = torch.randn(batch_size, 32)
    
    print("输入数据形状:")
    print(f"用户行为序列: {user_behavior_sequences.shape}")
    print(f"目标物品嵌入: {target_item_embeddings.shape}")
    print(f"行为类别: {behavior_categories.shape}")
    print(f"时间间隔: {time_intervals.shape}")
    print(f"用户特征: {user_features.shape}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        ctr_predictions, attention_weights, selected_indices = model(
            user_behavior_seq=user_behavior_sequences,
            target_item_emb=target_item_embeddings,
            user_features=user_features,
            behavior_categories=behavior_categories,
            time_intervals=time_intervals
        )
    
    print("输出数据形状:")
    print(f"预测CTR: {ctr_predictions.shape}")
    print(f"注意力权重: {attention_weights.shape}")
    print(f"选中行为索引: {selected_indices.shape}")
    
    print("基本功能测试完成")

def test_search_modes():
    """测试不同搜索模式"""
    print("\n开始测试不同搜索模式...")
    
    # 创建硬搜索模型
    hard_search_model = SIMModel(
        item_embedding_dim=32,
        user_feature_dim=16,
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        search_type='hard',
        num_categories=50
    )
    print("硬搜索模型创建成功")
    
    # 创建软搜索模型
    soft_search_model = SIMModel(
        item_embedding_dim=32,
        user_feature_dim=16,
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        search_type='soft',
        num_categories=50
    )
    print("软搜索模型创建成功")
    
    # 创建测试数据
    batch_size = 1
    seq_len = 50
    embedding_dim = 32
    
    user_behavior_sequences = torch.randn(batch_size, seq_len, embedding_dim)
    target_item_embeddings = torch.randn(batch_size, embedding_dim)
    behavior_categories = torch.randint(0, 50, (batch_size, seq_len))
    time_intervals = torch.randint(0, 100, (batch_size, 20))
    user_features = torch.randn(batch_size, 16)
    
    # 测试硬搜索
    with torch.no_grad():
        hard_search_output, _, _ = hard_search_model(
            user_behavior_seq=user_behavior_sequences,
            target_item_emb=target_item_embeddings,
            user_features=user_features,
            behavior_categories=behavior_categories,
            time_intervals=time_intervals
        )
    
    # 测试软搜索
    with torch.no_grad():
        soft_search_output, _, _ = soft_search_model(
            user_behavior_seq=user_behavior_sequences,
            target_item_emb=target_item_embeddings,
            user_features=user_features,
            time_intervals=time_intervals
        )
    
    print(f"硬搜索预测结果: {hard_search_output.item():.4f}")
    print(f"软搜索预测结果: {soft_search_output.item():.4f}")
    
    print("搜索模式测试完成")

if __name__ == "__main__":
    test_model_basic_functionality()
    test_search_modes()
    print("\n所有测试完成！")
