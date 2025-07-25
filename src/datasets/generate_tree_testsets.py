import torch
import torch_geometric
import networkx as nx
import os
import sys
from tqdm import tqdm

# ==================== 用户配置 (需要您修改) ====================
# 1. 项目 'src' 目录的绝对或相对路径
#    目的是让脚本能找到并导入项目中的 SpectreGraphDataset 类
PROJECT_SRC_PATH = ".."  # <--- 如果脚本不在项目根目录，请修改为正确的路径

# 2. 数据集的根目录路径，用于创建 'processed' 子目录并保存文件
#    这里以 'tree' 数据集为例
DATASET_ROOT_DIR = "../../data/my_tree"  # <--- 修改为您的目标数据集根目录

# 3. 数据集大小配置
TRAIN_GRAPHS = 128  # 训练集图的数量
VAL_GRAPHS = 32     # 验证集图的数量
TEST_GRAPHS = 100    # 测试集图的数量
# ==============================================================

# --- 将项目src目录添加到Python路径中 ---
try:
    # 确保路径存在
    if not os.path.isdir(PROJECT_SRC_PATH):
        raise FileNotFoundError
    sys.path.append(PROJECT_SRC_PATH)
    from datasets.spectre_dataset import SpectreGraphDataset

    print("✅ 成功从项目中导入 SpectreGraphDataset。")
except (ImportError, FileNotFoundError):
    print(f"❌ 错误: 无法在 '{PROJECT_SRC_PATH}' 找到 'datasets.spectre_dataset'。")
    print("请确保 PROJECT_SRC_PATH 指向了您项目的 'src' 目录。")
    exit()


def convert_nx_to_pyg_data(graph: nx.Graph) -> torch_geometric.data.Data:
    """
    将单个 networkx 图对象转换为 PyTorch Geometric 的 Data 对象。
    这个逻辑与 spectre_dataset.py 中的处理方式保持一致，以确保格式兼容。
    """
    # 从 networkx 图获取邻接矩阵
    adj = torch.Tensor(nx.to_numpy_array(graph))

    n = adj.shape[-1]

    # 节点特征：全为1，维度为 [N, 1]
    x = torch.ones(n, 1, dtype=torch.float)

    # 标签：空，维度为 [1, 0]
    y = torch.zeros([1, 0]).float()

    # 从邻接矩阵获取稀疏的边索引
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)

    # 边特征：对于存在的边，值为 [0, 1]，维度为 [num_edges, 2]
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1

    # 节点数量
    num_nodes = torch.tensor(n, dtype=torch.long).view(1)

    data = torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data


def generate_and_process_datasets(
    node_counts, 
    train_size=TRAIN_GRAPHS, 
    val_size=VAL_GRAPHS, 
    test_size=TEST_GRAPHS, 
    root_dir=DATASET_ROOT_DIR
):
    """
    为指定的节点数列表生成训练集、验证集和测试集，并直接处理成项目所需的格式。
    
    参数:
    - node_counts: 每个图中节点的数量列表
    - train_size: 训练集中每种节点数图的数量
    - val_size: 验证集中每种节点数图的数量
    - test_size: 测试集中每种节点数图的数量
    - root_dir: 保存数据集的根目录
    """
    raw_dir = os.path.join(root_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"所有处理好的数据集文件将保存在 '{raw_dir}/' 目录下。")

    # 创建一个临时的 Dataset 实例来调用它的 collate 方法
    class DummyDataset:
        def collate(self, data_list):
            return torch_geometric.data.Batch.from_data_list(data_list), None

    dummy_dataset = DummyDataset()

    # 为每种数据集(训练、验证、测试)生成数据
    datasets = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    for dataset_name, dataset_size in datasets.items():
        print(f"\n>>> 正在生成{dataset_name}集...")
        all_data_list = []
        
        for n_nodes in node_counts:
            print(f"  正在为 {n_nodes} 个节点的图生成 {dataset_name} 集...")
            
            for _ in tqdm(range(dataset_size), desc=f"  生成 {n_nodes}-node 图"):
                # 1. 生成一个随机树
                if n_nodes > 1:
                    nx_graph = nx.random_tree(n=n_nodes, seed=None)
                else:
                    nx_graph = nx.empty_graph(n=1)

                # 2. 转换为 PyG Data 对象
                pyg_data = convert_nx_to_pyg_data(nx_graph)
                all_data_list.append(pyg_data)
        
        # 3. 使用项目自带的 collate 方法打包数据列表
        collated_data, slices = dummy_dataset.collate(all_data_list)
        
        # 4. 定义文件名并保存打包好的数据
        file_path = os.path.join(raw_dir, f"{dataset_name}.pt")
        torch.save((collated_data, slices), file_path)
        
        print(f"  ✅ 已成功生成并打包 {len(all_data_list)} 个图，保存到 '{file_path}'")


if __name__ == "__main__":
    # 这里可以设置要生成的图的节点数
    # 为了简化，这里只使用单一节点数64，可以根据需要添加更多节点数
    node_list = [80]
    
    generate_and_process_datasets(
        node_counts=node_list,
        train_size=TRAIN_GRAPHS,
        val_size=VAL_GRAPHS,
        test_size=TEST_GRAPHS
    )
    
    print(f"\n🎉 训练集、验证集和测试集已成功生成！")
    print(f"📊 数据集统计:")
    print(f"   - 训练集: {len(node_list) * TRAIN_GRAPHS} 张图")
    print(f"   - 验证集: {len(node_list) * VAL_GRAPHS} 张图")
    print(f"   - 测试集: {len(node_list) * TEST_GRAPHS} 张图")
    print(f"⚠️  请确保数据已正确生成，接下来可以开始训练模型。") 