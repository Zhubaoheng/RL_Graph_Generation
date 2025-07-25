# generate_planar_dataset.py
import torch
import torch_geometric
import networkx as nx
import os
import sys
from tqdm import tqdm
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.data import Data, Batch
import torch_geometric.utils

# ==================== 用户配置 (需要您修改) ====================
# 1. 项目 'src' 目录的绝对或相对路径
#    目的是让脚本能找到并导入项目中的 SpectreGraphDataset 类
PROJECT_SRC_PATH = "/home/ly/max/DeFoG-swanlab/src"  # <--- 如果脚本不在项目根目录，请修改为正确的路径

# 2. 数据集的根目录路径，用于创建 'raw' 子目录并保存文件
#    这里以 'planar' 数据集为例，与 spectre_dataset.py 兼容
DATASET_ROOT_DIR = "/home/ly/max/DeFoG-swanlab/data/my_planar"  # <--- 修改为您的目标数据集根目录

# 3. 数据集大小配置
TRAIN_GRAPHS = 128  # 训练集图的数量
VAL_GRAPHS = 32     # 验证集图的数量
TEST_GRAPHS = 40   # 测试集图的数量
NUM_NODES = 128      # Planar数据集的节点数固定为64
# ==============================================================

# --- 将项目src目录添加到Python路径中 ---
try:
    # 确保路径存在
    if not os.path.isdir(PROJECT_SRC_PATH):
        raise FileNotFoundError
    sys.path.append(PROJECT_SRC_PATH)

    print("✅ 成功从项目中导入 SpectreGraphDataset。")
except (ImportError, FileNotFoundError):
    print(f"❌ 错误: 无法在 '{PROJECT_SRC_PATH}' 找到 'datasets.spectre_dataset'。")
    print("请确保 PROJECT_SRC_PATH 指向了您项目的 'src' 目录。")
    exit()

def generate_connected_planar_graph(num_nodes: int) -> nx.Graph:
    """使用Delaunay三角剖分生成一个连通的平面图。"""
    while True:
        pos = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
        points = np.array(list(pos.values()))
        delaunay_tri = Delaunay(points)
        graph = nx.Graph()
        for simplex in delaunay_tri.simplices:
            nx.add_cycle(graph, simplex)
        graph.add_nodes_from(range(num_nodes))
        if nx.is_connected(graph) and len(graph.nodes) == num_nodes:
            return graph

def convert_nx_to_adjacency_matrix(graph: nx.Graph) -> torch.Tensor:
    """
    将单个 networkx 图对象转换为邻接矩阵。
    这个格式与 spectre_dataset.py 期望的格式一致。
    """
    # 从 networkx 图获取邻接矩阵
    adj = torch.Tensor(nx.to_numpy_array(graph))
    return adj

def generate_and_save_datasets(
    node_counts, 
    train_size=TRAIN_GRAPHS, 
    val_size=VAL_GRAPHS, 
    test_size=TEST_GRAPHS, 
    root_dir=DATASET_ROOT_DIR
):
    """
    生成训练集、验证集和测试集，保存为邻接矩阵列表格式。
    这个格式与 spectre_dataset.py 期望的格式完全一致。
    
    参数:
    - node_counts: 每个图中节点的数量列表
    - train_size: 训练集中每种节点数图的数量
    - val_size: 验证集中每种节点数图的数量
    - test_size: 测试集中每种节点数图的数量
    - root_dir: 保存数据集的根目录
    """
    raw_dir = os.path.join(root_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"邻接矩阵数据文件将保存在 '{raw_dir}/' 目录下。")

    # 为每种数据集(训练、验证、测试)生成数据
    datasets = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    for dataset_name, dataset_size in datasets.items():
        print(f"\n>>> 正在生成{dataset_name}集...")
        adjacency_matrices = []
        
        for n_nodes in node_counts:
            print(f"  正在为 {n_nodes} 个节点的图生成 {dataset_name} 集...")
            
            for _ in tqdm(range(dataset_size), desc=f"  生成 {n_nodes}-node 平面图"):
                # 1. 生成一个连通的平面图
                nx_graph = generate_connected_planar_graph(n_nodes)

                # 2. 转换为邻接矩阵（与 spectre_dataset.py 期望的格式一致）
                adj_matrix = convert_nx_to_adjacency_matrix(nx_graph)
                adjacency_matrices.append(adj_matrix)
        
        # 3. 保存为邻接矩阵列表（与 spectre_dataset.py 期望的格式一致）
        file_path = os.path.join(raw_dir, f"{dataset_name}.pt")
        torch.save(adjacency_matrices, file_path)
        
        print(f"  ✅ 已成功生成并保存 {len(adjacency_matrices)} 个邻接矩阵到 '{file_path}'")

if __name__ == "__main__":
    # 这里设置要生成的图的节点数
    # Planar数据集使用固定的64个节点
    node_list = [NUM_NODES]
    
    print("🚀 开始生成 Planar 数据集（与 spectre_dataset.py 兼容的格式）...")
    generate_and_save_datasets(
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
    print(f"⚠️  数据格式与 spectre_dataset.py 完全兼容，可以直接使用 dataset=planar 运行。")