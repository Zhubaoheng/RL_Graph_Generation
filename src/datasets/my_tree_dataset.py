import os
import pathlib
import pickle

import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, Data

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class MyTreeGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='test',  # 'train', 'val', 或 'test'
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def processed_file_names(self):
        return [f"data_{self.split}.pt"]

    def download(self):
        # 无需下载，因为我们使用本地生成的数据
        pass

    def process(self):
        # 从原始文件加载数据
        raw_path = os.path.join(self.raw_dir, f"{self.split}.pt")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"未找到数据文件 {raw_path}，请先运行生成数据集的脚本")
            
        loaded_data = torch.load(raw_path)
        
        # 检查数据格式
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            data, slices = loaded_data
            
            # 如果slices是None，说明这是一个Batch格式的数据，需要转换
            if slices is None and hasattr(data, 'to_data_list'):
                print(f"Converting Batch data with {data.num_graphs} graphs...")
                # data是一个Batch对象，需要将其分割成单独的图
                data_list = data.to_data_list()
                
                # 使用标准的collate方法重新打包
                data, slices = self.collate(data_list)
                print(f"Conversion complete. Number of graphs: {len(slices['x']) - 1}")
            elif slices is not None:
                print(f"Data already in correct format with {len(slices['x']) - 1} graphs")
        else:
            # 如果是单个Batch对象
            if hasattr(loaded_data, 'to_data_list'):
                print(f"Converting single Batch data with {loaded_data.num_graphs} graphs...")
                data_list = loaded_data.to_data_list()
                data, slices = self.collate(data_list)
                print(f"Conversion complete. Number of graphs: {len(slices['x']) - 1}")
            else:
                print("Unknown data format, using as-is")
                data = loaded_data
                slices = None

        # 将数据保存为 PyG 格式
        torch.save((data, slices), self.processed_paths[0])


class MyTreeGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # 加载训练集、验证集和测试集
        train_dataset = MyTreeGraphDataset(root=root_path, split='train')
        val_dataset = MyTreeGraphDataset(root=root_path, split='val')
        test_dataset = MyTreeGraphDataset(root=root_path, split='test')
        
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset  # 使用训练集作为内部数据集


class MyTreeDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.dataset_name = "my_tree"  #  新的数据集名称
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


#  如果需要在外部转换 nx.Graph 到 PyG Data,  保留这个函数
def convert_nx_to_pyg_data(graph: nx.Graph) -> Data:
    """
    将单个 networkx 图对象转换为 PyTorch Geometric 的 Data 对象。
    这个逻辑与 generate_tree_testsets.py 中的处理方式保持一致，以确保格式兼容。
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

    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data 