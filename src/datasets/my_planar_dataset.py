# my_planar_dataset.py
import os
import pathlib
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, Data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import networkx as nx

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# --- 完全模仿 spectre_dataset.py 的结构和逻辑 ---

class MyPlanarGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='test',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        # 加载 process() 方法处理后的最终标准数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # process() 方法会从 raw 文件夹读取这个文件
        return [f"{self.split}.pt"]

    @property
    def processed_file_names(self):
        # process() 方法处理完后，会保存为这个文件名
        return [f"data_{self.split}.pt"]

    def download(self):
        # 无需下载，因为我们使用本地生成的数据
        pass

    def process(self):
        """
        参考 spectre_dataset.py 中的 process() 方法处理数据。
        读取原始数据，创建 PyG Data 对象，然后使用 collate 方法保存。
        """
        # 从原始文件加载数据
        raw_path = os.path.join(self.raw_dir, f"{self.split}.pt")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"未找到数据文件 {raw_path}，请先运行生成数据集的脚本")
            
        loaded_data = torch.load(raw_path)
        
        # 处理不同的数据格式
        if isinstance(loaded_data, list):
            # 如果是邻接矩阵列表格式（官方数据集格式），与 spectre_dataset.py 处理方式一致
            data_list = []
            for adj in loaded_data:
                n = adj.shape[-1]
                X = torch.ones(n, 1, dtype=torch.float)
                y = torch.zeros([1, 0]).float()
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)
                
                data_obj = Data(
                    x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                )
                
                if self.pre_filter is not None and not self.pre_filter(data_obj):
                    continue
                if self.pre_transform is not None:
                    data_obj = self.pre_transform(data_obj)
                
                data_list.append(data_obj)
        elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            # 如果是 (data, slices) 格式，需要转换回 data_list
            data, slices = loaded_data
            if slices is not None:
                # 从 collated data 中提取单个图的数据
                data_list = []
                num_graphs = len(slices['x']) - 1
                for i in range(num_graphs):
                    start_idx = slices['x'][i]
                    end_idx = slices['x'][i + 1]
                    
                    # 提取节点特征
                    x = data.x[start_idx:end_idx]
                    n = x.shape[0]
                    
                    # 重建邻接矩阵
                    edge_start = slices['edge_index'][i]
                    edge_end = slices['edge_index'][i + 1]
                    edges = data.edge_index[:, edge_start:edge_end] - start_idx
                    
                    # 创建邻接矩阵
                    adj = torch.zeros(n, n)
                    adj[edges[0], edges[1]] = 1
                    
                    # 按照 spectre_dataset 的方式重新创建 Data 对象
                    X = torch.ones(n, 1, dtype=torch.float)
                    y = torch.zeros([1, 0]).float()
                    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                    edge_attr[:, 1] = 1
                    num_nodes = n * torch.ones(1, dtype=torch.long)
                    
                    data_obj = Data(
                        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                    )
                    
                    if self.pre_filter is not None and not self.pre_filter(data_obj):
                        continue
                    if self.pre_transform is not None:
                        data_obj = self.pre_transform(data_obj)
                    
                    data_list.append(data_obj)
            else:
                # 如果是 Batch 格式，转换为 data_list
                if hasattr(data, 'to_data_list'):
                    original_data_list = data.to_data_list()
                    data_list = []
                    
                    for data_obj in original_data_list:
                        # 按照 spectre_dataset 的方式重新创建 Data 对象
                        n = data_obj.x.shape[0]
                        
                        # 重建邻接矩阵
                        adj = torch.zeros(n, n)
                        adj[data_obj.edge_index[0], data_obj.edge_index[1]] = 1
                        
                        X = torch.ones(n, 1, dtype=torch.float)
                        y = torch.zeros([1, 0]).float()
                        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                        edge_attr[:, 1] = 1
                        num_nodes = n * torch.ones(1, dtype=torch.long)
                        
                        new_data_obj = Data(
                            x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                        )
                        
                        if self.pre_filter is not None and not self.pre_filter(new_data_obj):
                            continue
                        if self.pre_transform is not None:
                            new_data_obj = self.pre_transform(new_data_obj)
                        
                        data_list.append(new_data_obj)
                else:
                    raise ValueError(f"Unknown data format: {type(data)}")
        else:
            # 如果是单个 Batch 对象
            if hasattr(loaded_data, 'to_data_list'):
                original_data_list = loaded_data.to_data_list()
                data_list = []
                
                for data_obj in original_data_list:
                    # 按照 spectre_dataset 的方式重新创建 Data 对象
                    n = data_obj.x.shape[0]
                    
                    # 重建邻接矩阵
                    adj = torch.zeros(n, n)
                    adj[data_obj.edge_index[0], data_obj.edge_index[1]] = 1
                    
                    X = torch.ones(n, 1, dtype=torch.float)
                    y = torch.zeros([1, 0]).float()
                    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                    edge_attr[:, 1] = 1
                    num_nodes = n * torch.ones(1, dtype=torch.long)
                    
                    new_data_obj = Data(
                        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
                    )
                    
                    if self.pre_filter is not None and not self.pre_filter(new_data_obj):
                        continue
                    if self.pre_transform is not None:
                        new_data_obj = self.pre_transform(new_data_obj)
                    
                    data_list.append(new_data_obj)
            else:
                raise ValueError(f"Unknown data format: {type(loaded_data)}")

        # 使用 collate 方法保存数据（与 spectre_dataset.py 一致）
        torch.save(self.collate(data_list), self.processed_paths[0])


class MyPlanarGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # 加载训练集、验证集和测试集
        train_dataset = MyPlanarGraphDataset(root=root_path, split='train')
        val_dataset = MyPlanarGraphDataset(root=root_path, split='val')
        test_dataset = MyPlanarGraphDataset(root=root_path, split='test')
        
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        
        # 打印数据集大小信息（与 spectre_dataset.py 一致）
        train_len = len(datasets["train"].data.n_nodes)
        val_len = len(datasets["val"].data.n_nodes)
        test_len = len(datasets["test"].data.n_nodes)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset  # 使用训练集作为内部数据集

    def __getitem__(self, item):
        return self.inner[item]


class MyPlanarDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.dataset_name = "my_planar"  # 新的数据集名称
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


#  如果需要在外部转换 nx.Graph 到 PyG Data,  保留这个函数
def convert_nx_to_pyg_data(graph: nx.Graph) -> Data:
    """
    将单个 networkx 图对象转换为 PyTorch Geometric 的 Data 对象。
    这个逻辑与 spectre_dataset.py 中的处理方式保持一致，以确保格式兼容。
    """
    # 从 networkx 图获取邻接矩阵
    adj = torch.Tensor(nx.to_numpy_array(graph))

    n = adj.shape[-1]

    # 按照 spectre_dataset.py 的标准格式创建 Data 对象
    X = torch.ones(n, 1, dtype=torch.float)
    y = torch.zeros([1, 0]).float()
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1
    num_nodes = n * torch.ones(1, dtype=torch.long)

    data = Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
    )
    return data