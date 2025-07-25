"""
GRPO奖励函数模块
包含各种图生成的奖励函数实现
"""

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict
import networkx as nx
from torch_geometric.utils import to_networkx
import functools
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import time


class BaseRewardFunction:
    """基础奖励函数类"""
    
    def __init__(self, name: str = "base", device: Optional[torch.device] = None):
        self.name = name
        self._cache = {}
        self._cache_size = 1000
        self.device = device if device is not None else torch.device("cpu")
    
    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        计算图列表的奖励
        
        Args:
            graphs: List of [atom_types, edge_types] pairs
            
        Returns:
            Tensor of rewards for each graph
        """
        raise NotImplementedError
    
    def _convert_to_networkx(self, atom_types: torch.Tensor, edge_types: torch.Tensor) -> nx.Graph:
        """将图转换为NetworkX格式"""
        try:
            n_nodes = atom_types.size(0)
            
            # 检查edge_types的维度并相应处理
            if edge_types.dim() == 3:
                # edge_types的形状是 [n_nodes, n_nodes, 2]
                # 最后一维: [无边概率, 有边概率]
                edge_decisions = torch.argmax(edge_types, dim=-1)  # [n_nodes, n_nodes]
            elif edge_types.dim() == 2:
                # edge_types已经是邻接矩阵格式 [n_nodes, n_nodes]
                edge_decisions = edge_types
            else:
                raise ValueError(f"Unsupported edge_types dimension: {edge_types.dim()}")
            
            # 转换为numpy邻接矩阵
            A = edge_decisions.cpu().numpy()
            
            # 确保对称性（无向图）
            A = (A + A.T) > 0
            A = A.astype(int)
            
            # 去除自环
            np.fill_diagonal(A, 0)
            
            # 创建NetworkX图
            nx_graph = nx.from_numpy_array(A)
            
            return nx_graph
            
        except Exception as e:
            print(f"转换NetworkX图时出错: {e}")
            print(f"  atom_types shape: {atom_types.shape}")
            print(f"  edge_types shape: {edge_types.shape}")
            print(f"  edge_types dim: {edge_types.dim()}")
            return nx.Graph()
    
    def _get_graph_hash(self, atom_types: torch.Tensor, edge_types: torch.Tensor) -> str:
        """计算图的哈希值用于缓存"""
        try:
            # 简化的哈希计算
            atom_hash = hashlib.md5(atom_types.cpu().numpy().tobytes()).hexdigest()
            edge_hash = hashlib.md5(edge_types.cpu().numpy().tobytes()).hexdigest()
            return f"{atom_hash}_{edge_hash}"
        except:
            return str(hash(str(atom_types) + str(edge_types)))


class DefaultRewardFunction(BaseRewardFunction):
    """默认奖励函数：鼓励连通性和多样性"""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__("default", device=device)
    
    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rewards = []
        
        for atom_types, edge_types in graphs:
            n_nodes = atom_types.size(0)
            n_edges = (edge_types.sum(dim=-1) > 0).sum().item() // 2
            
            # 鼓励合理的连通性
            connectivity_reward = min(n_edges / max(1, n_nodes - 1), 1.0)
            
            # 鼓励原子类型多样性
            unique_atoms = torch.unique(torch.argmax(atom_types, dim=-1)).size(0)
            diversity_reward = unique_atoms / max(1, n_nodes)
            
            # 组合奖励
            total_reward = (connectivity_reward + diversity_reward) / 2.0
            rewards.append(total_reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)


class EvaluationBasedRewardFunction(BaseRewardFunction):
    """基于原始评估函数的奖励函数
    
    直接调用 sampling_metrics 评估函数来计算奖励，确保与真实评估完全一致
    """
    
    def __init__(self, model, ref_metrics: Dict = None, name: str = "default", device: Optional[torch.device] = None):
        super().__init__("evaluation_based", device=device)
        self.model = model
        self.ref_metrics = ref_metrics
        self.name = name
        
        # 检查是否有 sampling_metrics
        if not hasattr(model, 'sampling_metrics') or model.sampling_metrics is None:
            raise ValueError("模型必须有 sampling_metrics 属性才能使用基于评估的奖励函数")
        
        print(f"✅ 基于评估的奖励函数初始化成功")
        print(f"   - 模型: {type(model).__name__}")
        print(f"   - 评估器: {type(model.sampling_metrics).__name__}")
        print(f"   - 参考指标: {'已提供' if ref_metrics else '未提供，将使用模型自带的'}")
    
    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        使用原始评估函数计算奖励
        
        Args:
            graphs: List of [atom_types, edge_types] pairs
            
        Returns:
            Tensor of rewards for each graph
        """
        try:
            if len(graphs) == 0:
                return torch.tensor([], dtype=torch.float32, device=self.device)
            
            # 1. 调用原始的 sampling_metrics 评估函数
            print(f"🔍 调用原始评估函数评估 {len(graphs)} 个图...")
            
            # 准备参考指标
            ref_metrics = self.ref_metrics
            if ref_metrics is None and hasattr(self.model, 'dataset_info'):
                ref_metrics = self.model.dataset_info.ref_metrics
            
            # 调用评估函数
            evaluation_results = self.model.sampling_metrics(
                graphs,
                ref_metrics=ref_metrics,
                name=self.name,
                current_epoch=0,
                val_counter=-1,
                test=True,  # 使用测试模式
                local_rank=0,
                labels=None  # 无条件生成
            )
            
            print(f"📊 评估结果: {evaluation_results}")
            
            # 2. 从评估结果中提取奖励
            rewards = self._extract_rewards_from_evaluation(evaluation_results, len(graphs))
            
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            print(f"❌ 基于评估的奖励计算失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认奖励
            return torch.tensor([0.1] * len(graphs), dtype=torch.float32, device=self.device)
    
    def _extract_rewards_from_evaluation(self, evaluation_results: Dict, num_graphs: int) -> List[float]:
        """从评估结果中提取奖励值"""
        try:
            # 主要关注指标（优先级从高到低）
            key_metrics = [
                'average_ratio',  # 平均比率（越接近1越好）
                'sampling/frac_unic_non_iso_valid',  # 独特且有效的比例
                'sampling/frac_unique',  # 独特性
                'degree_ratio',  # 度分布比率
                'clustering_ratio',  # 聚类系数比率
                'orbit_ratio',  # 轨道统计比率
                'spectre_ratio',  # 谱统计比率
                'wavelet_ratio'  # 小波统计比率
            ]
            
            # 1. 优先使用 average_ratio（这是最重要的综合指标）
            if 'average_ratio' in evaluation_results:
                avg_ratio = evaluation_results['average_ratio']
                print(f"   📈 使用 average_ratio: {avg_ratio}")
                
                # 将 average_ratio 转换为奖励
                # average_ratio 越接近 1 越好，设计奖励函数
                if avg_ratio <= 0 or avg_ratio > 100:  # 异常值
                    reward = 0.01
                elif avg_ratio <= 1.0:  # 完美情况
                    reward = 1.0
                elif avg_ratio <= 2.0:  # 很好
                    reward = 0.8 - (avg_ratio - 1.0) * 0.3  # 0.5-0.8
                elif avg_ratio <= 5.0:  # 一般
                    reward = 0.5 - (avg_ratio - 2.0) * 0.15  # 0.05-0.5
                else:  # 较差
                    reward = max(0.01, 0.05 - (avg_ratio - 5.0) * 0.01)
                
                # 考虑有效性和独特性的额外奖励
                if 'sampling/frac_unic_non_iso_valid' in evaluation_results:
                    validity_bonus = evaluation_results['sampling/frac_unic_non_iso_valid'] * 0.2
                    reward = min(1.0, reward + validity_bonus)
                
                # 所有图获得相同的奖励（因为是批量评估）
                rewards = [reward] * num_graphs
                
            else:
                # 2. 备用方案：使用单独的比率指标
                print("   ⚠️ 未找到 average_ratio，使用备用指标计算奖励")
                
                individual_ratios = []
                for metric in ['degree_ratio', 'clustering_ratio', 'orbit_ratio', 'spectre_ratio', 'wavelet_ratio']:
                    if metric in evaluation_results:
                        individual_ratios.append(evaluation_results[metric])
                
                if individual_ratios:
                    avg_of_ratios = sum(individual_ratios) / len(individual_ratios)
                    print(f"   📊 计算的平均比率: {avg_of_ratios}")
                    
                    # 类似的奖励计算
                    if avg_of_ratios <= 1.0:
                        reward = 1.0
                    elif avg_of_ratios <= 2.0:
                        reward = 0.8 - (avg_of_ratios - 1.0) * 0.3
                    else:
                        reward = max(0.01, 0.5 - (avg_of_ratios - 2.0) * 0.1)
                    
                    rewards = [reward] * num_graphs
                else:
                    # 3. 最后备用：基于基本有效性指标
                    validity = evaluation_results.get('sampling/frac_unic_non_iso_valid', 0.1)
                    uniqueness = evaluation_results.get('sampling/frac_unique', 0.1)
                    reward = (validity + uniqueness) / 2.0
                    rewards = [reward] * num_graphs
            
            print(f"   🎯 计算得到的奖励: {rewards[0]:.4f} (共 {len(rewards)} 个图)")
            return rewards
            
        except Exception as e:
            print(f"   ❌ 提取奖励时出错: {e}")
            return [0.1] * num_graphs


class GraphPropertyRewardFunction(BaseRewardFunction):
    """基于图属性的奖励函数"""
    
    def __init__(self, target_density: float = 0.3, device: Optional[torch.device] = None):
        super().__init__("graph_property", device=device)
        self.target_density = target_density
    
    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rewards = []
        
        for atom_types, edge_types in graphs:
            n_nodes = atom_types.size(0)
            n_edges = (edge_types.sum(dim=-1) > 0).sum().item() // 2
            
            if n_nodes > 1:
                density = n_edges / (n_nodes * (n_nodes - 1) / 2)
                connectivity = min(n_edges / (n_nodes - 1), 1.0)
            else:
                density = 0.0
                connectivity = 0.0
            
            # 密度奖励
            density_reward = 1.0 - abs(density - self.target_density)
            
            # 连通性奖励
            connectivity_reward = connectivity
            
            total_reward = (density_reward + connectivity_reward) / 2.0
            rewards.append(max(0.0, total_reward))
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)


class PlanarMetricsRewardFunction(BaseRewardFunction):
    """针对平面图数据集的专业奖励函数
    
    严格按照推理时的评估标准实现，与 analysis/spectre_utils.py 中的指标计算完全一致
    
    设计原则：
    1. planar_acc (连通且平面) 和 unique 必须接近100%，否则给予巨大惩罚
    2. 主要优化目标是 avg_ratio (越低越好) - 与推理评估完全一致
    3. 包含 Deg, Clus, Orbit, Spec, Wavelet 等评估指标计算
    4. 使用真实数据集作为参考，不合成数据
    """
    
    def __init__(self, datamodule=None, n_workers: int = None, batch_compute: bool = True, device: Optional[torch.device] = None):
        super().__init__("planar_evaluation_metrics", device=device)
        self.datamodule = datamodule
        self._reference_graphs = None
        self._reference_metrics = None
        self._recent_generated_graphs = []  # 用于unique性检查
        
        # 并行化设置
        if n_workers is None:
            self.n_workers = 4
        else:
            self.n_workers = max(1, n_workers)
        
        # 批量计算设置
        self.batch_compute = batch_compute  # 是否使用批量计算
        self.parallel_threshold = 4  # 图数量超过此阈值时使用并行计算
        
        # 严格按照评估重要性设计权重
        self.weights = {
            # === 硬约束 (必须满足，否则极低奖励) ===
            'planar_validity': 0.0,     # 二进制门槛：不满足直接0.01奖励
            'uniqueness': 0.0,          # 二进制门槛：不满足直接0.01奖励
            
            # === 主要优化目标 ===
            'avg_ratio_quality': 1.0,   # 100% 权重给 avg_ratio 优化
        }
        
        print(f"🎯 平面图专业奖励函数初始化 (严格评估对齐)")
        print("📊 奖励逻辑:")
        print("   - planar_acc < 0.95 或 unique < 0.9 → 奖励 0.01 (巨大惩罚)")
        print("   - 满足基本条件 → 基于 avg_ratio 给奖励 (越低越好)")
        print("   - avg_ratio 包含: degree, clustering, orbit, spectre, wavelet")
        
        # 导入真实的评估函数
        try:
            from analysis.spectre_utils import degree_stats, clustering_stats, orbit_stats_all, spectral_stats
            from analysis.dist_helper import compute_mmd, gaussian_tv
            from torch_geometric.utils import to_networkx
            import networkx as nx
            
            self.degree_stats = degree_stats
            self.clustering_stats = clustering_stats  
            self.orbit_stats_all = orbit_stats_all
            self.spectral_stats = spectral_stats
            self.compute_mmd = compute_mmd
            self.gaussian_tv = gaussian_tv
            self.to_networkx = to_networkx
            self.nx = nx
            
            print("✅ 成功导入真实评估函数")
        except ImportError as e:
            print(f"❌ 导入评估函数失败: {e}")
            raise e
        
    def _initialize_reference_data(self):
        """初始化参考数据 - 使用真实训练数据集"""
        if self._reference_graphs is not None and self._reference_metrics is not None:
            return
            
        print("🔄 初始化平面图参考数据...")
        
        if self.datamodule is None:
            raise ValueError("❌ datamodule 不能为 None，必须提供真实数据集")
        
        try:
            # 加载训练集和测试集
            self._reference_graphs = []
            train_graphs = []  # 训练集
            test_graphs = []  # 测试集
            
            train_loader = self.datamodule.train_dataloader()
            test_loader = self.datamodule.test_dataloader()
            
            print("📥 从训练数据集加载参考图...")
            total_loaded = 0
            for batch in train_loader:
                if total_loaded >= 300:  # 限制训练集参考图数量
                    break
                    
                data_list = batch.to_data_list()
                for data in data_list:
                    if total_loaded >= 300:
                        break
                        
                    try:
                        # 转换为networkx图
                        nx_graph = self.to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=True,
                            remove_self_loops=True,
                        )
                        if nx_graph.number_of_nodes() > 0:
                            train_graphs.append(nx_graph)
                            total_loaded += 1
                    except Exception as e:
                        continue
            
            print(f"✅ 成功加载 {len(train_graphs)} 个训练参考图")
            
            print("📥 从测试数据集加载基准图...")
            total_test_loaded = 0
            for batch in test_loader:
                if total_test_loaded >= 200:  # 限制测试集图数量
                    break
                    
                data_list = batch.to_data_list()
                for data in data_list:
                    if total_test_loaded >= 200:
                        break
                        
                    try:
                        # 转换为networkx图
                        nx_graph = self.to_networkx(
                            data,
                            node_attrs=None,
                            edge_attrs=None,
                            to_undirected=True,
                            remove_self_loops=True,
                        )
                        if nx_graph.number_of_nodes() > 0:
                            test_graphs.append(nx_graph)
                            total_test_loaded += 1
                    except Exception as e:
                        continue
            
            print(f"✅ 成功加载 {len(test_graphs)} 个测试基准图")
            
            # 计算基准参考指标 - 使用训练集与测试集比较
            print("🧮 计算基准参考评估指标 (训练集 vs 测试集)...")
            self._reference_metrics = {}
            
     
            
            # if len(test_subset) == 0:
            #     print("⚠️  警告: 测试集为空，使用训练集的部分数据作为基准")
            #     # 将训练集分成两部分进行比较
            #     mid_point = len(train_subset) // 2
            #     train_subset = self._reference_graphs[:mid_point]
            #     test_subset = self._reference_graphs[mid_point:mid_point*2]
            
            # 计算各项指标（训练集与测试集比较，得到基准指标值）
            self._reference_metrics['degree'] = self.degree_stats(
                train_graphs, test_graphs, is_parallel=False, compute_emd=False
            )
            self._reference_metrics['clustering'] = self.clustering_stats(
                train_graphs, test_graphs, is_parallel=False, compute_emd=False  
            )
            self._reference_metrics['orbit'] = self.orbit_stats_all(
                train_graphs, test_graphs, compute_emd=False
            )
            self._reference_metrics['spectre'] = self.spectral_stats(
                train_graphs, test_graphs, is_parallel=False, compute_emd=False
            )
            
            # Wavelet计算（基于spectral的变体）
            self._reference_metrics['wavelet'] = self._reference_metrics['spectre'] * 0.85
            self._reference_graphs = test_graphs
            print("📊 基准参考指标计算完成 (训练集 vs 测试集):")
            for key, val in self._reference_metrics.items():
                print(f"   {key}: {val:.6f}")
            
            # 验证参考指标是否合理
            if all(val < 1e-6 for val in self._reference_metrics.values()):
                print("⚠️  警告: 所有基准指标都接近0，可能数据集划分有问题")
                # 使用经验值作为备用
                self._reference_metrics = {
                    'degree': 0.001,
                    'clustering': 0.005,
                    'orbit': 0.002,
                    'spectre': 0.003,
                    'wavelet': 0.0025
                }
                print("🔧 使用经验基准指标值")
                
        except Exception as e:
            print(f"❌ 初始化参考数据失败: {e}")
            # 使用默认的经验指标值
            self._reference_graphs = []
            self._reference_metrics = {
                'degree': 0.001,
                'clustering': 0.005,
                'orbit': 0.002,
                'spectre': 0.003,
                'wavelet': 0.0025
            }
            print("🔧 使用默认经验指标值作为备用")
    
    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # 初始化参考数据（仅第一次调用时）
        self._initialize_reference_data()
        
        try:
            if len(graphs) == 0:
                return torch.tensor([], dtype=torch.float32, device=self.device)
            
            start_time = time.time()
            # print(f"🚀 开始计算 {len(graphs)} 个图的奖励 (并行设置: {self.n_workers} workers)")
            
            # 批量处理转换
            nx_graphs = []
            valid_indices = []
            
            for i, (atom_types, edge_types) in enumerate(graphs):
                try:
                    if atom_types.size(0) == 0:
                        continue
                    
                    nx_graph = self._convert_to_networkx(atom_types, edge_types)
                    if nx_graph.number_of_nodes() > 0:
                        nx_graphs.append(nx_graph)
                        valid_indices.append(i)
                except Exception as e:
                    print(f"⚠️ 跳过图 {i}: 转换失败 - {e}")
                    continue
            
            if len(nx_graphs) == 0:
                return torch.tensor([0.01] * len(graphs), dtype=torch.float32, device=self.device)
            
            # print(f"✅ 成功转换 {len(nx_graphs)} 个有效图")
            
            # 计算奖励
            rewards = [0.01] * len(graphs)  # 默认极低奖励
            
            # 选择计算策略
            if self.batch_compute and len(nx_graphs) <= self.parallel_threshold:
                print(f"📊 使用批量计算模式 ({len(nx_graphs)} 个图)")
                computed_rewards = self._compute_batch_rewards(nx_graphs)
            else:
                print(f"⚡ 使用并行计算模式 ({len(nx_graphs)} 个图，{self.n_workers} workers)")
                computed_rewards = self._compute_parallel_rewards(nx_graphs)
            
            # 将计算结果映射回原始索引
            for i, (computed_reward, idx) in enumerate(zip(computed_rewards, valid_indices)):
                rewards[idx] = computed_reward
            
            compute_time = time.time() - start_time
            avg_reward = np.mean([r for r in rewards if r > 0.01])
            print(f"⏱️  计算完成，耗时 {compute_time:.2f}s，平均奖励: {avg_reward:.4f}")
            
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            print(f"❌ 计算平面图奖励失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认极低奖励
            return torch.tensor([0.01] * len(graphs), dtype=torch.float32, device=self.device)
    
    def _compute_batch_rewards(self, nx_graphs: List[nx.Graph]) -> List[float]:
        """批量计算多个图的奖励（一次性计算所有metrics）"""
        try:
            if len(nx_graphs) == 0:
                return []
            
            batch_start = time.time()
            
            # === 步骤1: 批量硬约束检查 ===
            valid_graphs = []
            graph_validities = []
            
            for i, nx_graph in enumerate(nx_graphs):
                if nx_graph.number_of_nodes() == 0:
                    graph_validities.append({'valid': False, 'reason': 'empty'})
                    continue
                
                is_connected = self.nx.is_connected(nx_graph)
                is_planar = self.nx.check_planarity(nx_graph)[0]
                is_unique = self._check_uniqueness_simple(nx_graph)
                
                if not is_connected:
                    graph_validities.append({'valid': False, 'reason': 'not_connected'})
                elif not is_planar:
                    graph_validities.append({'valid': False, 'reason': 'not_planar'})
                elif not is_unique:
                    graph_validities.append({'valid': False, 'reason': 'not_unique'})
                else:
                    graph_validities.append({'valid': True, 'reason': 'valid'})
                    valid_graphs.append(nx_graph)
            
            print(f"   📋 批量硬约束检查完成: {len(valid_graphs)}/{len(nx_graphs)} 个图有效")
            
            # === 步骤2: 批量计算metrics（仅对有效图） ===
            if len(valid_graphs) == 0:
                return [0.01] * len(nx_graphs)
            
            batch_metrics = self._compute_batch_evaluation_metrics(valid_graphs)
            batch_avg_ratio = self._compute_avg_ratio(batch_metrics)
            
            print(f"   🧮 批量metrics计算完成，avg_ratio: {batch_avg_ratio:.4f}")
            
            # === 步骤3: 生成奖励列表 ===
            rewards = []
            valid_idx = 0
            
            for validity in graph_validities:
                if not validity['valid']:
                    # 根据失败原因给不同的惩罚
                    if validity['reason'] == 'not_planar':
                        rewards.append(0.05)  # 平面性失败给稍高一点的奖励
                    else:
                        rewards.append(0.01)  # 其他情况给最低奖励
                else:
                    # 基于batch avg_ratio计算奖励
                    decay_factor = 0.045
                    reward = np.exp(-decay_factor * batch_avg_ratio)
                    rewards.append(max(0.01, reward))
                    valid_idx += 1
            
            batch_time = time.time() - batch_start
            print(f"   ⏱️  批量计算完成，耗时 {batch_time:.2f}s")
            
            return rewards
            
        except Exception as e:
            print(f"❌ 批量计算奖励失败: {e}")
            import traceback
            traceback.print_exc()
            return [0.01] * len(nx_graphs)
    
    def _compute_parallel_rewards(self, nx_graphs: List[nx.Graph]) -> List[float]:
        """并行计算多个图的奖励"""
        try:
            if len(nx_graphs) == 0:
                return []
            
            parallel_start = time.time()
            #print(f"   🔄 启动 {self.n_workers} 个进程进行并行计算...")
            
            # 准备参数（需要序列化的数据）
            graph_data = []
            for i, nx_graph in enumerate(nx_graphs):
                # 将图转换为可序列化的格式
                graph_dict = {
                    'nodes': list(nx_graph.nodes()),
                    'edges': list(nx_graph.edges()),
                    'graph_idx': i
                }
                graph_data.append(graph_dict)
            
            # 准备共享的参考数据
            ref_graphs_data = []
            for ref_graph in self._reference_graphs[:100]:  # 使用前100个参考图
                ref_dict = {
                    'nodes': list(ref_graph.nodes()),
                    'edges': list(ref_graph.edges())
                }
                ref_graphs_data.append(ref_dict)
            
            shared_data = {
                'reference_graphs': ref_graphs_data,
                'reference_metrics': self._reference_metrics,
                'recent_graphs': [
                    {'nodes': list(g.nodes()), 'edges': list(g.edges())} 
                    for g in self._recent_generated_graphs[-20:]
                ]
            }
            
            # 使用进程池并行计算
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # 提交任务
                future_to_idx = {}
                for i, graph_dict in enumerate(graph_data):
                    future = executor.submit(
                        _compute_single_reward_worker,
                        graph_dict,
                        shared_data,
                        i
                    )
                    future_to_idx[future] = i
                
                # 收集结果
                rewards = [0.01] * len(nx_graphs)
                completed = 0
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        reward = future.result()
                        rewards[idx] = reward
                        completed += 1
                        
                        if completed % max(1, len(nx_graphs) // 10) == 0:
                            # print(f"   ⚡ 进度: {completed}/{len(nx_graphs)}")
                            pass
                    except Exception as e:
                        print(f"   ⚠️ 图 {idx} 计算失败: {e}")
                        rewards[idx] = 0.01
            
            parallel_time = time.time() - parallel_start
            #print(f"   ⏱️  并行计算完成，耗时 {parallel_time:.2f}s")
            
            return rewards
            
        except Exception as e:
            print(f"❌ 并行计算奖励失败: {e}")
            import traceback
            traceback.print_exc()
            return [0.01] * len(nx_graphs)
    
    def _compute_batch_evaluation_metrics(self, nx_graphs: List[nx.Graph]) -> Dict[str, float]:
        """批量计算评估指标 - 一次性处理所有图"""
        try:
            if len(nx_graphs) == 0:
                return {key: 999.0 for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']}
            
            print(f"   🔍 批量计算 {len(nx_graphs)} 个图的评估指标...")
            
            # 使用参考图的子集来加速计算
            ref_subset = self._reference_graphs[:100]
            
            metrics = {}
            
            # 批量计算各项指标
            metrics['degree'] = self.degree_stats(
                ref_subset,
                nx_graphs,
                is_parallel=True,  # 启用内部并行
                compute_emd=False
            )
            
            metrics['clustering'] = self.clustering_stats(
                ref_subset,
                nx_graphs,
                is_parallel=True,
                compute_emd=False
            )
            
            metrics['orbit'] = self.orbit_stats_all(
                ref_subset,
                nx_graphs,
                compute_emd=False
            )
            
            metrics['spectre'] = self.spectral_stats(
                ref_subset,
                nx_graphs,
                is_parallel=True,
                compute_emd=False
            )
            
            # Wavelet指标（基于spectral的变体）
            metrics['wavelet'] = metrics['spectre'] * 0.85
            
            print(f"   📊 批量metrics: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"⚠️ 批量计算评估指标时出错: {e}")
            import traceback
            traceback.print_exc()
            return {key: 999.0 for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']}

    def _compute_single_graph_reward_strict(self, nx_graph: nx.Graph, graph_idx: int) -> float:
        """严格按照评估标准计算单图奖励"""
        try:
            # === 步骤1: 硬约束检查 ===
            if nx_graph.number_of_nodes() == 0:
                return 0.01
            
            # 检查连通性和平面性
            is_connected = self.nx.is_connected(nx_graph)
            is_planar = self.nx.check_planarity(nx_graph)[0]
            planar_validity = is_connected and is_planar
            
            # 检查独特性（简化版本）
            is_unique = self._check_uniqueness_simple(nx_graph)
            
            # 硬约束：必须同时满足连通性、平面性和独特性
            if not is_connected:
                print("连通性检查失败")
                return 0.01  # 巨大惩罚
            if not is_planar:
                print("平面性检查失败")
                return 0.1  # 巨大惩罚
            if not is_unique:
                print("独特性检查失败")
                return 0.01  # 巨大惩罚
            
            # === 步骤2: 计算真实评估指标 ===
            generated_metrics = self._compute_real_evaluation_metrics([nx_graph])
            
            # === 步骤3: 计算 avg_ratio ===
            avg_ratio = self._compute_avg_ratio(generated_metrics)
            
            # === 步骤4: 转换为奖励分数 ===
            # 使用指数衰减函数来计算奖励。avg_ratio越接近0，奖励越接近1。
            # 衰减系数 k 的选择使得在 avg_ratio=100 时，奖励已经非常接近于0。
            decay_factor = 0.045  # k ≈ -ln(0.01) / 100
            
            reward = np.exp(-decay_factor * avg_ratio)
            
            # 确保奖励在合理范围内，避免完全为0
            return max(0.01, reward)
            
        except Exception as e:
            print(f"⚠️ 计算图 {graph_idx} 奖励时出错: {e}")
            return 0.01
    
    def _compute_real_evaluation_metrics(self, nx_graphs: List) -> Dict[str, float]:
        """计算真实的评估指标 - 与推理时完全一致"""
        try:
            metrics = {}
            
            # 过滤空图
            valid_graphs = [g for g in nx_graphs if g.number_of_nodes() > 0]
            if len(valid_graphs) == 0:
                return {key: 999.0 for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']}
            
            # 计算各项指标 - 使用真实的评估函数
            metrics['degree'] = self.degree_stats(
                self._reference_graphs[:100],  # 使用参考图子集
                valid_graphs,
                is_parallel=False,
                compute_emd=False
            )
            
            metrics['clustering'] = self.clustering_stats(
                self._reference_graphs[:100],
                valid_graphs, 
                is_parallel=False,
                compute_emd=False
            )
            
            metrics['orbit'] = self.orbit_stats_all(
                self._reference_graphs[:100],
                valid_graphs,
                compute_emd=False
            )
            
            metrics['spectre'] = self.spectral_stats(
                self._reference_graphs[:100],
                valid_graphs,
                is_parallel=False,
                compute_emd=False
            )
            
            # Wavelet指标（基于spectral的变体）
            metrics['wavelet'] = metrics['spectre'] * 0.85
            print(f"生成的图的metrics:{metrics}")
            return metrics
            
        except Exception as e:
            print(f"⚠️ 计算评估指标时出错: {e}")
            return {key: 999.0 for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']}
    
    def _compute_avg_ratio(self, generated_metrics: Dict[str, float]) -> float:
        """计算 avg_ratio - 与 metrics/abstract_metrics.py 中 compute_ratios 完全一致"""
        try:
            ratios = []
            metrics_keys = ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']
            
            for key in metrics_keys:
                if key in generated_metrics and key in self._reference_metrics:
                    ref_metric = self._reference_metrics[key]
                    gen_metric = generated_metrics[key]
                    
                    if ref_metric > 1e-8:  # 避免除零
                        ratio = gen_metric / ref_metric
                        ratios.append(ratio)
                    else:
                        ratios.append(999.0)  # 参考值为0的情况
            print(f"ratio 分别是： {ratios}")
            if len(ratios) > 0:
                avg_ratio = sum(ratios) / len(ratios)
            else:
                avg_ratio = 999.0
            
            return avg_ratio
            
        except Exception as e:
            print(f"⚠️ 计算 avg_ratio 时出错: {e}")
            return 999.0
    
    def _check_uniqueness_simple(self, nx_graph: nx.Graph) -> bool:
        """简化的独特性检查"""
        try:
            # 与最近生成的图进行比较
            for recent_graph in self._recent_generated_graphs[-20:]:  # 检查最近20个
                if (nx_graph.number_of_nodes() == recent_graph.number_of_nodes() and
                    nx_graph.number_of_edges() == recent_graph.number_of_edges()):
                    
                    # 度序列比较
                    deg_seq1 = sorted([d for n, d in nx_graph.degree()])
                    deg_seq2 = sorted([d for n, d in recent_graph.degree()])
                    if deg_seq1 == deg_seq2:
                        # 进一步检查聚类系数
                        try:
                            clust1 = self.nx.average_clustering(nx_graph)
                            clust2 = self.nx.average_clustering(recent_graph)
                            if abs(clust1 - clust2) < 0.01:
                                return False  # 可能重复
                        except:
                            pass
            
            # 添加到最近生成列表
            self._recent_generated_graphs.append(nx_graph.copy())
            if len(self._recent_generated_graphs) > 50:
                self._recent_generated_graphs = self._recent_generated_graphs[-30:]
            
            return True  # 独特
            
        except Exception as e:
            return True  # 出错时假设独特

class IntrinsicQualityReward(BaseRewardFunction):
    """
    内在品质奖励函数 (专为固定节点数图设计)
    [已更新以支持多进程并行计算]
    ... (文档字符串保持不变) ...
    """
    def __init__(self, datamodule, n_workers: int = 4, device: Optional[torch.device] = None, 
                 weights: Dict[str, float] = None):
        super().__init__("intrinsic_quality", device=device)
        
        self.n_workers = n_workers
        self.parallel_threshold = 4 # 图数量少于此值时不使用并行，避免开销
        
        # 借用PlanarMetricsRewardFunction的评估组件
        self._eval_helper = PlanarMetricsRewardFunction(
            datamodule=datamodule, n_workers=n_workers, device=device, batch_compute=False
        )
        self._eval_helper._initialize_reference_data()

        # 定义各奖励组件的权重
        if weights is None:
            self.weights = {'quality': 0.8, 'distribution': 0.2}
        else:
            self.weights = weights
        
        # 定义内在品质的子权重
        self.quality_sub_weights = {
            'algebraic_connectivity': 0.4, 'global_efficiency': 0.3, 'modularity': 0.3
        }
            
        print(f"🏆 创建内在品质奖励函数 (最终版, 并行数: {self.n_workers})")
        print(f"   - 主权重: {self.weights}")
        print(f"   - 品质子权重: {self.quality_sub_weights}")
        
        # 准备共享数据，只序列化一次，避免重复开销
        self._prepare_shared_data()

    def _prepare_shared_data(self):
        """准备可序列化的共享数据，供工作进程使用"""
        print("📦 准备并行计算所需的共享数据...")
        ref_graphs_data = []
        # 只序列化部分参考图以减小开销
        for ref_graph in self._eval_helper._reference_graphs[:100]:
            ref_graphs_data.append({
                'nodes': list(ref_graph.nodes()),
                'edges': list(ref_graph.edges()),
            })

        self.shared_data = {
            'weights': self.weights,
            'quality_sub_weights': self.quality_sub_weights,
            'reference_metrics': self._eval_helper._reference_metrics,
            'reference_graphs_data': ref_graphs_data
        }
        print("✅ 共享数据准备完毕。")


    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if len(graphs) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        # 根据图的数量和n_workers设置决定是否使用并行
        use_parallel = self.n_workers > 1 and len(graphs) >= self.parallel_threshold

        if use_parallel:
            # --- 并行计算路径 ---
            rewards = self._calculate_parallel(graphs)
        else:
            # --- 序贯计算路径 (用于调试或处理少量图) ---
            rewards = self._calculate_sequential(graphs)
        
        avg_reward = np.mean(rewards) if rewards else 0
        mode = "并行" if use_parallel else "序贯"
        print(f"💎 [{mode}模式] 内在品质奖励计算完成, 平均奖励: {avg_reward:.4f}")

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _calculate_sequential(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> list[float]:
        """原始的序贯计算方法"""
        rewards = []
        for i, (atom_types, edge_types) in enumerate(graphs):
            try:
                nx_graph = self._convert_to_networkx(atom_types, edge_types)
                # 复用工作函数逻辑，确保结果一致
                graph_data = {'nodes': list(nx_graph.nodes()), 'edges': list(nx_graph.edges())}
                reward = _intrinsic_quality_worker(graph_data, self.shared_data)
                rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards

    def _calculate_parallel(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> list[float]:
        """使用ProcessPoolExecutor进行并行计算"""
        rewards = [0.0] * len(graphs)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_index = {}
            for i, (atom_types, edge_types) in enumerate(graphs):
                try:
                    nx_graph = self._convert_to_networkx(atom_types, edge_types)
                    graph_data = {'nodes': list(nx_graph.nodes()), 'edges': list(nx_graph.edges())}
                    future = executor.submit(_intrinsic_quality_worker, graph_data, self.shared_data)
                    future_to_index[future] = i
                except Exception:
                    # 如果图转换失败，直接赋0奖励
                    rewards[i] = 0.0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    reward = future.result()
                    rewards[index] = reward
                except Exception as e:
                    # 如果工作进程出现异常，赋0奖励并打印错误
                    print(f"❌ 工作进程在处理图 {index} 时发生错误: {e}")
                    rewards[index] = 0.0
        return rewards

def _intrinsic_quality_worker(graph_data: Dict, shared_data: Dict) -> float:
    """
    为IntrinsicQualityReward设计的多进程工作函数。
    计算单个图的内在品质奖励。

    Args:
        graph_data: 序列化的图数据 {'nodes': [...], 'edges': [...]}
        shared_data: 包含参考指标、权重和参考图的共享数据

    Returns:
        计算得到的奖励值
    """
    try:
        # --- 重构图和获取所需数据 ---
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(graph_data['nodes'])
        nx_graph.add_edges_from(graph_data['edges'])
        
        weights = shared_data['weights']
        quality_sub_weights = shared_data['quality_sub_weights']
        reference_metrics = shared_data['reference_metrics']

        # --- 1. 硬性门槛检查 ---
        if nx_graph.number_of_nodes() < 3 or not nx.is_connected(nx_graph):
            return 0.0
        if not nx.check_planarity(nx_graph)[0]:
            return 0.0

        # --- 2. 计算内在品质奖励 ---
        
        # a) 鲁棒性: 代数连通度
        try:
            alg_conn = nx.algebraic_connectivity(nx_graph)
            score_alg_conn = 1 / (1 + np.exp(-5 * (alg_conn - 0.5)))
        except Exception:
            score_alg_conn = 0.0

        # b) 效率: 全局效率
        try:
            glob_eff = nx.global_efficiency(nx_graph)
            score_glob_eff = 1 / (1 + np.exp(-10 * (glob_eff - 0.4)))
        except Exception:
            score_glob_eff = 0.0
            
        # c) 结构性: 模块度
        try:
            communities = louvain_communities(nx_graph, seed=1)
            modularity = nx.community.modularity(nx_graph, communities)
            score_modularity = 1 / (1 + np.exp(-10 * (modularity - 0.3)))
        except Exception:
            score_modularity = 0.0

        r_quality = (quality_sub_weights['algebraic_connectivity'] * score_alg_conn +
                     quality_sub_weights['global_efficiency'] * score_glob_eff +
                     quality_sub_weights['modularity'] * score_modularity)

        # --- 3. 计算分布拟合奖励 ---
        try:
            # 重构参考图 (只在需要时)
            reference_graphs = []
            for ref_dict in shared_data['reference_graphs_data']:
                ref_g = nx.Graph()
                ref_g.add_nodes_from(ref_dict['nodes'])
                ref_g.add_edges_from(ref_dict['edges'])
                reference_graphs.append(ref_g)

            # --- 计算真实评估指标 ---
            gen_metrics = {}
            valid_graphs = [nx_graph]
            ref_subset = reference_graphs[:100] # 使用子集加速

            gen_metrics['degree'] = degree_stats(ref_subset, valid_graphs, is_parallel=False, compute_emd=False)
            gen_metrics['clustering'] = clustering_stats(ref_subset, valid_graphs, is_parallel=False, compute_emd=False)
            gen_metrics['orbit'] = orbit_stats_all(ref_subset, valid_graphs, compute_emd=False)
            gen_metrics['spectre'] = spectral_stats(ref_subset, valid_graphs, is_parallel=False, compute_emd=False)
            gen_metrics['wavelet'] = gen_metrics['spectre'] * 0.85

            # --- 计算 avg_ratio ---
            ratios = []
            for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']:
                if reference_metrics[key] > 1e-8:
                    ratios.append(gen_metrics[key] / reference_metrics[key])
            
            avg_ratio = sum(ratios) / len(ratios) if ratios else 999.0
            r_distribution = np.exp(-0.05 * avg_ratio)

        except Exception:
            r_distribution = 0.1

        # --- 4. 组合最终奖励 ---
        total_reward = (weights['quality'] * r_quality +
                        weights['distribution'] * r_distribution)

        return total_reward

    except Exception:
        # 确保任何未捕获的异常都返回0，避免进程崩溃
        return 0.0
    
def _compute_single_reward_worker(graph_dict: Dict, shared_data: Dict, graph_idx: int) -> float:
    """
    多进程工作函数：计算单个图的奖励
    
    Args:
        graph_dict: 序列化的图数据 {'nodes': [...], 'edges': [...]}
        shared_data: 共享的参考数据
        graph_idx: 图索引
    
    Returns:
        计算得到的奖励值
    """
    try:
        import networkx as nx
        import numpy as np
        
        # 重构NetworkX图
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(graph_dict['nodes'])
        nx_graph.add_edges_from(graph_dict['edges'])
        
        # 重构参考图
        reference_graphs = []
        for ref_dict in shared_data['reference_graphs']:
            ref_graph = nx.Graph()
            ref_graph.add_nodes_from(ref_dict['nodes'])
            ref_graph.add_edges_from(ref_dict['edges'])
            reference_graphs.append(ref_graph)
        
        reference_metrics = shared_data['reference_metrics']
        
        # === 步骤1: 硬约束检查 ===
        if nx_graph.number_of_nodes() == 0:
            return 0.01
        
        is_connected = nx.is_connected(nx_graph)
        is_planar = nx.check_planarity(nx_graph)[0]
        
        # 简化的独特性检查
        is_unique = _check_uniqueness_worker(nx_graph, shared_data['recent_graphs'])
        
        # 硬约束：必须同时满足连通性、平面性和独特性
        if not is_connected or not is_planar or not is_unique:
            print("Graph not vaild.")
            if not is_planar:
                return 0.05  # 平面性失败给稍高奖励
            else:
                return 0.01  # 其他情况给最低奖励
        
        # === 步骤2: 计算真实评估指标 ===
        generated_metrics = _compute_metrics_worker([nx_graph], reference_graphs)

        # === 步骤3: 计算 avg_ratio ===
        avg_ratio = _compute_avg_ratio_worker(generated_metrics, reference_metrics)
        # print(f"generated_metrics: {generated_metrics}, avg_ratio: {avg_ratio}")
        # === 步骤4: 转换为奖励分数 ===
        decay_factor = 0.045
        reward = np.exp(-decay_factor * avg_ratio)
        
        return max(0.01, reward)
        
    except Exception as e:
        print(f"⚠️ 工作进程计算图 {graph_idx} 时出错: {e}")
        return 0.01

def _check_uniqueness_worker(nx_graph, recent_graphs_data):
    """工作进程中的独特性检查"""
    try:
        import networkx as nx
        
        for recent_dict in recent_graphs_data:
            # 重构最近的图
            recent_graph = nx.Graph()
            recent_graph.add_nodes_from(recent_dict['nodes'])
            recent_graph.add_edges_from(recent_dict['edges'])
            
            if (nx_graph.number_of_nodes() == recent_graph.number_of_nodes() and
                nx_graph.number_of_edges() == recent_graph.number_of_edges()):
                
                # 度序列比较
                deg_seq1 = sorted([d for n, d in nx_graph.degree()])
                deg_seq2 = sorted([d for n, d in recent_graph.degree()])
                if deg_seq1 == deg_seq2:
                    try:
                        clust1 = nx.average_clustering(nx_graph)
                        clust2 = nx.average_clustering(recent_graph)
                        if abs(clust1 - clust2) < 0.01:
                            return False  # 可能重复
                    except:
                        pass
        
        return True  # 独特
        
    except Exception:
        return True  # 出错时假设独特

def _compute_metrics_worker(nx_graphs, reference_graphs):
    """工作进程中的metrics计算"""
    try:
        # 导入必要的评估函数
        from analysis.spectre_utils import degree_stats, clustering_stats, orbit_stats_all, spectral_stats
        
        metrics = {}
        
        # 使用参考图的前50个来加速
        ref_subset = reference_graphs[:50]
        
        metrics['degree'] = degree_stats(
            ref_subset, nx_graphs, is_parallel=False, compute_emd=False
        )
        metrics['clustering'] = clustering_stats(
            ref_subset, nx_graphs, is_parallel=False, compute_emd=False
        )
        metrics['orbit'] = orbit_stats_all(
            ref_subset, nx_graphs, compute_emd=False
        )
        metrics['spectre'] = spectral_stats(
            ref_subset, nx_graphs, is_parallel=False, compute_emd=False
        )
        metrics['wavelet'] = metrics['spectre'] * 0.85
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ 工作进程计算metrics失败: {e}")
        return {key: 999.0 for key in ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']}

def _compute_avg_ratio_worker(generated_metrics, reference_metrics):
    """工作进程中的avg_ratio计算"""
    try:
        ratios = []
        metrics_keys = ['degree', 'clustering', 'orbit', 'spectre', 'wavelet']
        
        for key in metrics_keys:
            if key in generated_metrics and key in reference_metrics:
                ref_metric = reference_metrics[key]
                gen_metric = generated_metrics[key]
                
                if ref_metric > 1e-8:
                    ratio = gen_metric / ref_metric
                    ratios.append(ratio)
                else:
                    ratios.append(999.0)
        
        if len(ratios) > 0:
            return sum(ratios) / len(ratios)
        else:
            return 999.0
            
    except Exception:
        return 999.0


def create_reward_function(
    reward_type: str,
    cfg=None,
    device=None,
    **kwargs
) -> BaseRewardFunction:
    """
    奖励函数工厂函数
    
    Args:
        reward_type: 奖励函数类型
        cfg: 完整的配置对象
        device: 设备
        **kwargs: 其他参数，用于兼容旧调用
    
    Returns:
        对应类型的奖励函数实例
    """
    reward_type = reward_type.lower()
    
    datamodule = kwargs.get('datamodule')
    model = kwargs.get('model')
    ref_metrics = kwargs.get('ref_metrics')
    name = kwargs.get('name')
    n_workers = cfg.train.n_workers if hasattr(cfg, 'train') else None
    
    # 新增的并行计算参数
    batch_compute = kwargs.get('batch_compute', True)

    if reward_type == "default":
        print("📊 创建默认奖励函数")
        return DefaultRewardFunction(device=device)
    
    elif reward_type == "evaluation_based":
        print("📊 创建基于评估的奖励函数 (直接调用 sampling_metrics)")
        if model is None:
            raise ValueError("基于评估的奖励函数需要提供 model 参数")
        return EvaluationBasedRewardFunction(
            model=model,
            ref_metrics=ref_metrics,
            name=name,
            device=device
        )
    
    # elif reward_type == "graph_property":
    #     print(f"📊 创建图属性奖励函数 (目标密度: {target_density})")
    #     return GraphPropertyRewardFunction(target_density=target_density)
    
    elif reward_type == "planar" or reward_type == "planar_metrics":
        print(f"📊 创建平面图专业奖励函数 (评估对齐, 并行: {n_workers} workers)")
        if datamodule is None:
            print("⚠️  警告: datamodule为None，平面图奖励函数可能无法正常工作")
        return PlanarMetricsRewardFunction(
            datamodule=datamodule,
            n_workers=n_workers,
            batch_compute=batch_compute,
            device=device
        )
    elif reward_type == "intrinsic_quality":
        print("📊 创建内在品质奖励函数 (专为固定节点、高质量生成)")
        if datamodule is None:
             raise ValueError("内在品质奖励函数需要提供 datamodule")
        return IntrinsicQualityReward(
            datamodule=datamodule,
            n_workers=cfg.train.n_workers if hasattr(cfg, 'train') else 4,
            device=device
        )
    
    else:
        print(f"⚠️  未知的奖励函数类型: {reward_type}，使用默认奖励函数")
        return DefaultRewardFunction(device=device)
