import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src import utils
from flow_matching import flow_matching_utils
from grpo_rewards import create_reward_function
from graph_discrete_flow_model import GraphDiscreteFlowModel


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) 训练器
    
    用于微调图生成模型的强化学习训练器。
    采用分组相对策略优化方法，结合PPO裁剪机制。
    """
    
    def __init__(
        self,
        model: pl.LightningModule,
        reward_function: Callable[[List], torch.Tensor],
        cfg: Dict,
        model_kwargs: dict,
    ):
        """
        初始化GRPO训练器
        
        Args:
            model: 预训练的图生成模型（可能被DataParallel包装）
            reward_function: 奖励函数，接收图列表返回奖励张量
            cfg: 配置字典
            model_kwargs: 创建新模型实例所需的参数
        """
        self.model = model
        self.reward_function = reward_function
        self.cfg = cfg
        self.model_kwargs = model_kwargs
        grpo_config = cfg.grpo
        
        # 从配置中提取参数
        self.learning_rate = grpo_config.learning_rate
        self.group_size = grpo_config.group_size
        self.num_groups = grpo_config.num_groups
        self.beta = grpo_config.kl_penalty # KL惩罚系数
        self.clip_ratio = grpo_config.clip_ratio # PPO裁剪比例
        self.gradient_accumulation_steps = grpo_config.gradient_accumulation_steps
        self.ref_model_update_freq = getattr(grpo_config, 'ref_model_update_freq', 200) # 参考模型更新频率
        
        # 检查模型是否被DataParallel包装
        self.is_multi_gpu = hasattr(model, 'module')
        
        # 获取原始模型（去除DataParallel包装）
        self.core_model = model.module if self.is_multi_gpu else model
        
        # 节点数配置
        self.target_node_count = getattr(grpo_config, 'target_node_count', None)
        self.node_count_range = None
        if hasattr(grpo_config, 'node_count_range') and grpo_config.node_count_range is not None:
            self.node_count_range = tuple(grpo_config.node_count_range)
        elif (hasattr(grpo_config, 'node_count_min') and hasattr(grpo_config, 'node_count_max') and
              grpo_config.node_count_min is not None and grpo_config.node_count_max is not None):
            self.node_count_range = (grpo_config.node_count_min, grpo_config.node_count_max)

        # 验证节点数配置
        self._validate_node_config()
        
        # 存储原始模型参数用于KL惩罚 - 将在 on_train_start 中初始化
        self.original_model_state = None
        
        # 创建参考策略模型用于重要性权重计算
        self.reference_model = None
        # 延迟创建reference model，等待模型设备分配完成
        self._reference_model_created = False
        
        # 训练步数计数器
        self.global_step = 0
        
        print(f"GRPO训练器初始化完成:")
        print(f" 组大小: {self.group_size}, 组数: {self.num_groups}")
        print(f" 学习率: {self.learning_rate}, KL惩罚: {self.beta}, PPO裁剪: {self.clip_ratio}")
        print(f" 目标节点数: {self.target_node_count}, 节点数范围: {self.node_count_range}")
        print(f" 多GPU模式: {self.is_multi_gpu}")

    def _validate_node_config(self):
        """验证节点数配置"""
        if self.target_node_count is None and self.node_count_range is None:
            raise ValueError(
                "必须指定节点数配置！请设置 target_node_count 或 node_count_range"
            )

    def _create_reference_model(self):
        """
        创建参考策略模型，适配DDP架构。
        DDP模式下结构更简单，无需复杂的分片处理。
        """
        print("🔄 创建参考策略模型 (DDP模式)...")
        
        # 获取主模型的设备
        device = next(self.model.parameters()).device
        
        # 步骤 1: 创建一个新的、普通的模型实例，作为参考模型的“容器”
        # 这个新模型在CPU上创建，以避免占用GPU显存
        self.reference_model = GraphDiscreteFlowModel(cfg=self.cfg, **self.model_kwargs).to('cpu')
        
        # 步骤 2: 获取DDP模型的状态字典
        # 🔧 修复：确保在获取state_dict前模型处于eval模式
        original_training_mode = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                # DDP模式下直接获取state_dict，无需特殊API
                full_state_dict = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}
        finally:
            # 恢复原始训练模式
            self.model.train(original_training_mode)

        # 步骤 3: 重映射状态字典的键以匹配参考模型
        # DDP 包装 GRPOLightningModule 后，参数键名可能包含:
        # `module.model.model.layer.weight` 或 `model.model.layer.weight`
        # 而我们的参考模型 (一个普通的 GraphDiscreteFlowModel) 期望的键名是:
        # `model.layer.weight`
        # 因此，我们需要剥离多余的前缀。
        print("    重映射state_dict键名以匹配参考模型结构...")
        remapped_state_dict = {}
        
        # DDP可能的前缀
        possible_prefixes = ["module.model.", "model."]
        
        # 检查收到的键，确定使用哪个前缀
        first_key = next(iter(full_state_dict.keys()), "")
        prefix_to_strip = ""
        
        for prefix in possible_prefixes:
            if any(k.startswith(prefix) for k in full_state_dict.keys()):
                prefix_to_strip = prefix
                break
        
        if not prefix_to_strip:
             print(f"    ⚠️ 警告: 在 state_dict 中未找到预期的前缀。第一个键是 '{first_key}'。")
             print(f"    ℹ️ 这可能意味着模型包装结构已更改。将尝试不剥离前缀。")

        for k, v in full_state_dict.items():
            if prefix_to_strip and k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
                remapped_state_dict[new_key] = v
            else:
                # 如果没有前缀，也保留该键（例如，非模型参数）
                 remapped_state_dict[k] = v
        
        # 步骤 4: 将提取并重映射后的完整权重加载到新的参考模型实例中
        self.reference_model.load_state_dict(remapped_state_dict, strict=True)
        # 步骤 5: 将参考模型移动到正确的GPU设备上，并设置为评估模式
        self.reference_model = self.reference_model.to(device)
        
        # 冻结参考模型参数
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.reference_model.eval()
        self._reference_model_created = True
        print("✅ 参考策略模型创建完成 (DDP模式)")

    def _ensure_reference_model(self):
        """确保参考模型已创建，如果没有则创建"""
        if not self._reference_model_created:
            self._create_reference_model()
        else:
            # 🔧 检查设备一致性，在DDP环境下设备可能会改变
            model_device = next(self.model.parameters()).device
            if self.reference_model is not None:
                ref_device = next(self.reference_model.parameters()).device
                if model_device != ref_device:
                    print(f"🔧 检测到设备不一致，移动参考模型从 {ref_device} 到 {model_device}")
                    self.reference_model = self.reference_model.to(model_device)

    def _update_reference_model(self, update_frequency: int = 1000):
        """
        更新参考策略模型，适配DDP架构。
        """
        import torch.distributed as dist

        # 确保参考模型已创建，这是先决条件
        self._ensure_reference_model()
        
        # 在第0步之后才开始更新，因为第0步时模型和参考模型是完全一样的
        if self.global_step > 0 and self.global_step % update_frequency == 0:
            print(f"🔄 更新参考策略模型 (step {self.global_step}) (DDP模式)")
            
            device = next(self.model.parameters()).device
            
            try:
                # 🔧 修复：确保在获取state_dict前模型处于eval模式，避免与梯度计算冲突
                original_training_mode = self.model.training
                self.model.eval()
                
                # 使用torch.no_grad确保没有梯度计算干扰
                with torch.no_grad():
                    # 步骤 1: 获取DDP模型的状态字典
                    # DDP模式下直接获取state_dict，无需特殊API
                    full_state_dict = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

                # 恢复原始训练模式
                self.model.train(original_training_mode)

                # 步骤 2: 使用与创建参考模型时完全相同的键重映射逻辑
                remapped_state_dict = {}
                # DDP可能的前缀
                possible_prefixes = ["module.model.", "model."]
                
                # 检查收到的键，确定使用哪个前缀
                first_key = next(iter(full_state_dict.keys()), "")
                prefix_to_strip = ""
                
                for prefix in possible_prefixes:
                    if any(k.startswith(prefix) for k in full_state_dict.keys()):
                        prefix_to_strip = prefix
                        break
                
                # 健壮性检查: 确认预期的前缀存在
                if not prefix_to_strip:
                     print(f"    ⚠️ 警告: 更新参考模型时，在 state_dict 中未找到预期的前缀。第一个键是 '{first_key}'。")
                     print(f"    ℹ️ 将尝试不剥离前缀直接加载。")

                for k, v in full_state_dict.items():
                    if prefix_to_strip and k.startswith(prefix_to_strip):
                        new_key = k[len(prefix_to_strip):]
                        remapped_state_dict[new_key] = v.to(device)  # 移动到正确设备
                    else:
                        remapped_state_dict[k] = v.to(device)
                
                # 步骤 3: 在no_grad环境下加载新状态到参考模型中
                with torch.no_grad():
                    self.reference_model.load_state_dict(remapped_state_dict, strict=True)
                print("    ✅ 参考模型权重更新成功 (DDP模式)。")
                
            except Exception as e:
                print(f"    ❌ 更新参考模型权重失败: {e}")
                print(f"    🔧 尝试的解决方案：跳过此次更新，继续训练")
                import traceback
                traceback.print_exc()
                return

            # 步骤 4: 确保参考模型在正确的设备上并处于评估模式
            self.reference_model = self.reference_model.to(device)
            self.reference_model.eval()

    def _sample_node_counts(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        根据配置采样节点数
        
        Args:
            batch_size: 批量大小
            device: 设备
            
        Returns:
            节点数张量 shape: (batch_size,)
        """
        if self.target_node_count is not None:
            # 固定节点数
            return torch.full((batch_size,), self.target_node_count, dtype=torch.long, device=device)
        elif self.node_count_range is not None:
            # 范围内随机采样
            min_nodes, max_nodes = self.node_count_range
            return torch.randint(min_nodes, max_nodes + 1, (batch_size,), dtype=torch.long, device=device)
        else:
            raise ValueError("节点数配置无效")

    def _run_model_forward(self, X_t, E_t, y_t, t, node_mask):
        """
        可被梯度检查点包装的模型前向传播函数。
        注意: 为了与 `grad_checkpoint` 兼容, 此函数只接受和返回张量。
        """
        # grad_checkpoint 在 `use_reentrant=False` 模式下不支持 None 输入,
        # 所以我们需要一个占位符。由于模型内部会处理条件/非条件情况，
        # 我们可以安全地传入一个零张量作为占位符。
        if y_t is None:
            # 创建一个与 X_t 设备和类型相匹配的虚拟张量
            y_t = torch.empty(X_t.size(0), 0, device=X_t.device, dtype=X_t.dtype)

        noisy_data = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
        extra_data = self.core_model.compute_extra_data(noisy_data)
        pred = self.core_model.forward(noisy_data, extra_data, node_mask)
        # grad_checkpoint 要求输出是张量元组
        return pred.X, pred.E, pred.y

    def _compute_policy_entropy(self, model_pred: utils.PlaceHolder, node_mask: torch.Tensor) -> torch.Tensor:
        """计算策略的熵，用于鼓励探索。"""
        # 节点熵
        X_logits = model_pred.X
        X_probs = F.softmax(X_logits, dim=-1)
        X_log_probs = F.log_softmax(X_logits, dim=-1)
        entropy_X = -(X_probs * X_log_probs).sum(dim=-1)
        # 为避免除以零，添加一个小的epsilon
        masked_entropy_X = (entropy_X * node_mask).sum() / (node_mask.sum() + 1e-8)

        # 边熵
        E_logits = model_pred.E
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        E_probs = F.softmax(E_logits, dim=-1)
        E_log_probs = F.log_softmax(E_logits, dim=-1)
        entropy_E = -(E_probs * E_log_probs).sum(dim=-1)
        # 为避免除以零，添加一个小的epsilon
        masked_entropy_E = (entropy_E * edge_mask).sum() / (edge_mask.sum() + 1e-8)
        
        return (masked_entropy_X + masked_entropy_E) / 2.0

    def sample_graphs_with_gradients(
        self,
        batch_size: int,
        num_nodes: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        total_inference_steps: int = 100,
    ) -> Tuple[utils.PlaceHolder, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        从完全噪声状态开始采样图，保持梯度计算并记录采样轨迹的对数概率。
        此版本完整复现了原始模型的高质量推理逻辑，并修复了内存泄漏问题。
        
        Args:
            batch_size: 批量大小
            num_nodes: 节点数张量 shape: (batch_size,)，如果为None则根据配置采样
            seed: 随机种子
            total_inference_steps: 总推理步数
        
        Returns:
            (生成的图, 节点掩码, 累积损失张量, 当前策略对数概率, 参考策略对数概率)
        """
        original_mode = self.model.training
        self.model.eval()
        try:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed % (2**31))
                random.seed(seed)
            
            device = next(self.model.parameters()).device
            
            # 步骤 1: 采样或确定节点数
            if num_nodes is None:
                n_nodes = self._sample_node_counts(batch_size, device)
            elif isinstance(num_nodes, int):
                n_nodes = num_nodes * torch.ones(batch_size, device=device, dtype=torch.int)
            else:
                n_nodes = num_nodes
            
            n_max = torch.max(n_nodes).item()
            
            # 步骤 2: 构建节点掩码
            arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
            node_mask = arange < n_nodes.unsqueeze(1)
            
            # 步骤 3: 从完全噪声状态 z_T 开始
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.core_model.noise_dist.get_limit_dist(),
                node_mask=node_mask
            )
            
            # 步骤 4: 处理条件标签 y (如果模型是条件模型)
            if self.core_model.conditional:
                # 修正：与原始 sample_batch 中的逻辑完全一致
                if "qm9" in self.cfg.dataset.name:
                    if hasattr(self.core_model, 'test_labels') and self.core_model.test_labels is not None:
                        y = self.core_model.test_labels
                        perm = torch.randperm(y.size(0))
                        idx = perm[:100]
                        condition = y[idx].to(device)
                        z_T.y = condition.repeat([10, 1])[:batch_size, :]
                    else:
                        # 如果没有test_labels，生成随机条件（用于调试）
                        z_T.y = torch.randn(batch_size, self.core_model.output_dims["y"]).to(device)
                elif "tls" in self.cfg.dataset.name:
                    z_T.y = torch.zeros(batch_size, 1).to(device)
                    z_T.y[:batch_size // 2] = 1
                else:
                    # 修正：与原始实现一致，对不支持的数据集抛出异常
                    # 这确保条件处理的严格一致性
                    raise NotImplementedError(f"Conditional sampling not implemented for dataset: {self.cfg.dataset.name}")
            
            # 步骤 5: 初始化状态和记录变量
            X, E, y = z_T.X, z_T.E, z_T.y
            
            # 🔧 保持完整的计算图，用于精确的梯度计算
            cumulative_loss = torch.tensor(0.0, device=device, requires_grad=True)
            current_log_probs = torch.zeros(batch_size, device=device, requires_grad=False)
            reference_log_probs = torch.zeros(batch_size, device=device, requires_grad=False)
            pred_final_step = None
            # --- 参考模型更新检查 ---
            # 检查当前是否为参考模型的更新步骤。DDP下此操作更简单，无需特殊处理。
            is_ref_update_step = (self.global_step > 0 and self.global_step % self.ref_model_update_freq == 0)
            if is_ref_update_step:
                print(f"   ℹ️ 步骤 {self.global_step}: 即将更新参考模型（DDP模式）。")
            # --- 检查结束 ---

            # 步骤 6: 核心推理循环
            for t_int in tqdm(range(total_inference_steps), desc="  ...采样轨迹", leave=False):
                # 计算当前时间步 t 和下一个时间步 s
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / (total_inference_steps + 1)
                
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    t_norm = t_norm + 1e-6
                
                s_array = t_array + 1
                s_norm = s_array / (total_inference_steps + 1)
                
                # 应用时间扭曲
                t_norm = self.core_model.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.core_model.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )
                
                # --- 在每个步骤计算中间损失和对数概率 ---
                with torch.enable_grad():
                    # 准备带梯度的数据
                    X_temp = X.detach().requires_grad_(True)
                    E_temp = E.detach().requires_grad_(True)
                    if y is not None:
                        if y.dtype.is_floating_point:
                            y_temp = y.detach().requires_grad_(True)
                        else:
                            y_temp = y.detach()
                    else:
                        y_temp = None

                    # 激活检查点已在GraphTransformer内部实现，此处直接调用即可
                    pred_X, pred_E, pred_y = self._run_model_forward(
                        X_temp, E_temp, y_temp, t_norm, node_mask
                    )
                    pred = utils.PlaceHolder(X=pred_X, E=pred_E, y=pred_y)
                    if t_int == total_inference_steps - 1:
                        pred_final_step = pred
                    # 计算并累积损失 (使用同一次前向传播的结果)
                    intermediate_loss = self._compute_intermediate_loss(X_temp, E_temp, pred, node_mask)
                    cumulative_loss = cumulative_loss + intermediate_loss
                    
                    # 计算并累积对数概率 (使用同一次前向传播的结果)
                    step_log_prob = self._compute_step_log_probability(X_temp, E_temp, pred, node_mask)
                    current_log_probs = current_log_probs + step_log_prob # 保持梯度连接
                    
                    # 参考策略 (无梯度)
                    with torch.no_grad():
                        self._ensure_reference_model()
                        ref_noisy_data = {"X_t": X, "E_t": E, "y_t": y, "t": t_norm, "node_mask": node_mask}
                        ref_extra_data = self.reference_model.compute_extra_data(ref_noisy_data)
                        ref_pred = self.reference_model.forward(ref_noisy_data, ref_extra_data, node_mask)
                        ref_step_log_prob = self._compute_step_log_probability(X, E, ref_pred, node_mask)
                        reference_log_probs = reference_log_probs + ref_step_log_prob.detach()

                # --- 核心采样逻辑：完整复现原始推理流程 ---
                if t_int < total_inference_steps - 1:
                    # 🔧 在采样过程中不保持梯度，因为采样操作本身是不可导的
                    with torch.no_grad():
                        noisy_data_no_grad = {
                            "X_t": X, "E_t": E, "y_t": y, "t": t_norm, "node_mask": node_mask
                        }
                        
                        # 1. 计算有条件预测
                        extra_data = self.core_model.compute_extra_data(noisy_data_no_grad)
                        pred = self.core_model.forward(noisy_data_no_grad, extra_data, node_mask)
                        pred_X_cond = F.softmax(pred.X, dim=-1)
                        pred_E_cond = F.softmax(pred.E, dim=-1)
                        
                        # 2. 计算有条件 rate matrix
                        G_1_pred_cond = (pred_X_cond, pred_E_cond)
                        G_t = (X, E)
                        R_t_X, R_t_E = self.core_model.rate_matrix_designer.compute_graph_rate_matrix(
                            t_norm, node_mask, G_t, G_1_pred_cond,
                        )
                        
                        # 3. 复现条件引导 (Conditional Guidance)
                        if self.core_model.conditional:
                            uncond_y = torch.ones_like(y, device=y.device) * -1
                            noisy_data_no_grad["y_t"] = uncond_y
                            
                            extra_data_uncond = self.core_model.compute_extra_data(noisy_data_no_grad)
                            uncond_pred = self.core_model.forward(noisy_data_no_grad, extra_data_uncond, node_mask)
                            uncond_pred_X = F.softmax(uncond_pred.X, dim=-1)
                            uncond_pred_E = F.softmax(uncond_pred.E, dim=-1)
                            
                            G_1_pred_uncond = (uncond_pred_X, uncond_pred_E)
                        
                            R_t_X_uncond, R_t_E_uncond = self.core_model.rate_matrix_designer.compute_graph_rate_matrix(
                                t_norm, node_mask, G_t, G_1_pred_uncond,
                            )
                            
                            # 在对数空间混合 rate matrices
                            guidance_weight = self.core_model.cfg.general.guidance_weight
                            R_t_X = torch.exp(
                                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight) +
                                torch.log(R_t_X + 1e-6) * guidance_weight
                            )
                            R_t_E = torch.exp(
                                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight) +
                                torch.log(R_t_E + 1e-6) * guidance_weight
                            )
                        
                        # 4. 复现转移概率计算 (调用模型内置函数)
                        dt = (s_norm - t_norm)[0]
                        prob_X, prob_E = self.core_model.compute_step_probs(
                            R_t_X, R_t_E, X, E, dt,
                            self.core_model.noise_dist.get_limit_dist().X,
                            self.core_model.noise_dist.get_limit_dist().E
                        )
                        
                        # 5. 复现最后一步的特殊处理
                        if s_norm[0] == 1.0:
                            prob_X, prob_E = pred_X_cond, pred_E_cond
                        
                        # 6. 使用计算出的概率进行采样
                        sampled_s = flow_matching_utils.sample_discrete_features(
                            prob_X, prob_E, node_mask=node_mask
                        )
                        
                        # 7. 更新状态为 one-hot 格式，用于下一次循环
                        X = F.one_hot(sampled_s.X, num_classes=X.size(-1)).float()
                        E = F.one_hot(sampled_s.E, num_classes=E.size(-1)).float()
                        
                        # 确保边矩阵的对称性
                        assert (E == torch.transpose(E, 1, 2)).all()

            # 步骤 8: 返回最终的、清理过的图

            # 修正：在返回前，移除虚拟类别，与原始实现(sample_batch)完全一致
            # 这是保证生成图结构正确的关键步骤
            X, E, y = self.core_model.noise_dist.ignore_virtual_classes(X, E, y)

            clean_graphs = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
            
            # 🔧 主动清理内存
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return clean_graphs, node_mask, cumulative_loss, current_log_probs, reference_log_probs, pred_final_step
        
        finally:
            self.model.train(original_mode)

    def _compute_intermediate_loss(
        self,
        X_t: torch.Tensor,  # shape: (batch_size, n_max, n_node_types)
        E_t: torch.Tensor,  # shape: (batch_size, n_max, n_max, n_edge_types)
        model_pred: utils.PlaceHolder,
        node_mask: torch.Tensor  # shape: (batch_size, n_max)
    ) -> torch.Tensor:
        """
        计算中间状态的损失。
        此版本经过重构，直接接收模型预测结果以支持梯度检查点。
        """
        # 接收模型预测
        pred = model_pred
        
        # 计算当前状态的负对数似然
        X_current = torch.argmax(X_t, dim=-1)  # shape: (batch_size, n_max)
        E_current = torch.argmax(E_t, dim=-1)  # shape: (batch_size, n_max, n_max)
        
        # 使用交叉熵损失
        nll_X = F.cross_entropy(
            pred.X.view(-1, pred.X.size(-1)),
            X_current.view(-1),
            reduction='none'
        ).view(X_current.shape)  # shape: (batch_size, n_max)
        
        nll_E = F.cross_entropy(
            pred.E.view(-1, pred.E.size(-1)),
            E_current.view(-1),
            reduction='none'
        ).view(E_current.shape)  # shape: (batch_size, n_max, n_max)
        
        # 应用掩码
        masked_nll_X = (nll_X * node_mask).sum()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # shape: (batch_size, n_max, n_max)
        masked_nll_E = (nll_E * edge_mask).sum()
        
        return masked_nll_X + masked_nll_E

    def _compute_step_log_probability(
        self,
        X_current: torch.Tensor,  # shape: (batch_size, n_max, n_node_types)
        E_current: torch.Tensor,  # shape: (batch_size, n_max, n_max, n_edge_types)
        model_pred: utils.PlaceHolder,
        node_mask: torch.Tensor,  # shape: (batch_size, n_max)
    ) -> torch.Tensor:
        """
        计算当前采样步骤的对数概率
        
        Returns:
            该步骤的对数概率 shape: (batch_size,)
        """
        batch_size = X_current.size(0)
        device = X_current.device
        
        # 计算节点转移的对数概率
        X_logits = model_pred.X
        X_probs = F.softmax(X_logits, dim=-1)  # shape: (batch_size, n_max, n_node_types)
        X_log_probs = torch.log(X_probs + 1e-8)
        
        # 获取实际采样的类别索引
        X_indices = torch.argmax(X_current, dim=-1)  # shape: (batch_size, n_max)
        
        # 收集对应的对数概率
        X_step_log_prob = torch.gather(X_log_probs, dim=-1, 
                                      index=X_indices.unsqueeze(-1)).squeeze(-1)
        
        # 应用节点掩码并求和
        X_masked_log_prob = (X_step_log_prob * node_mask).sum(dim=-1)  # shape: (batch_size,)
        
        # 计算边转移的对数概率
        E_logits = model_pred.E
        E_probs = F.softmax(E_logits, dim=-1)  # shape: (batch_size, n_max, n_max, n_edge_types)
        E_log_probs = torch.log(E_probs + 1e-8)
        
        # 获取边的类别索引
        E_indices = torch.argmax(E_current, dim=-1)  # shape: (batch_size, n_max, n_max)
        
        # 收集边的对数概率
        E_step_log_prob = torch.gather(E_log_probs, dim=-1,
                                      index=E_indices.unsqueeze(-1)).squeeze(-1)
        
        # 边掩码：只考虑有效的边
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # shape: (batch_size, n_max, n_max)
        E_masked_log_prob = (E_step_log_prob * edge_mask).sum(dim=(-2, -1))  # shape: (batch_size,)
        
        # 总的步骤对数概率
        total_step_log_prob = X_masked_log_prob + E_masked_log_prob
        
        return total_step_log_prob

    def _convert_to_graph_list(self, graphs: utils.PlaceHolder, node_mask: torch.Tensor) -> List:
        """将PlaceHolder图转换为列表格式"""
        graph_list = []
        X, E, y = graphs.X, graphs.E, graphs.y
        
        for i in range(X.size(0)):
            n_nodes = node_mask[i].sum().item()
            
            # 获取节点特征：转换为离散标签
            if X.dim() == 3:  # one-hot编码 (batch_size, n_nodes, n_node_types)
                atom_types = torch.argmax(X[i, :n_nodes], dim=-1)  # shape: (n_nodes,)
            else:  # 已经是离散标签
                atom_types = X[i, :n_nodes]
                
            # 获取边特征：转换为离散标签  
            if E.dim() == 4:  # one-hot编码 (batch_size, n_nodes, n_nodes, n_edge_types)
                edge_types = torch.argmax(E[i, :n_nodes, :n_nodes], dim=-1)  # shape: (n_nodes, n_nodes)
            else:  # 已经是离散标签
                edge_types = E[i, :n_nodes, :n_nodes]
            
            # 处理特殊情况：当只有一个节点时
            if n_nodes == 1:
                # atom_types已经是(1,)形状，无需调整
                # edge_types已经是(1,1)形状，无需调整
                pass
            
            # 确保数据类型正确
            if atom_types.dtype != torch.long:
                atom_types = atom_types.long()
            if edge_types.dtype != torch.long:
                edge_types = edge_types.long()
                
            graph_list.append([atom_types, edge_types])
        
        return graph_list

    def compute_grpo_loss(
        self,
        rewards: torch.Tensor,
        current_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        model_preds: utils.PlaceHolder,
        node_masks: torch.Tensor,
        global_rank: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算GRPO损失、KL散度以及策略熵奖励 (已适配DDP梯度流)。
        """
        import torch.distributed as dist
        device = rewards.device
        loss_metrics = {}

        # --- 安全检查 ---
        if rewards.numel() == 0:
            if global_rank == 0:
                print("⚠️ [Rank 0] GRPO损失计算: 输入的rewards张量为空，跳过。")
            return {"total_loss": torch.tensor(0.0, device=device, requires_grad=True)}

        # --- 全局基线和指标计算 ---
        # 由于数据已经全局同步，直接计算即可
        global_baseline_reward = rewards.mean()
        loss_metrics['reward/baseline_reward_global'] = global_baseline_reward.item()
        loss_metrics['reward/max_reward_global'] = rewards.max().item()
        loss_metrics['reward/min_reward_global'] = rewards.min().item()

        # --- 策略损失计算 ---
        # 裁剪 log_ratio 防止 exp() 溢出
        log_ratio = current_log_probs - reference_log_probs.detach()
        log_ratio = torch.clamp(log_ratio, min=-15.0, max=15.0)
        
        ratio = torch.exp(log_ratio)
        advantage = rewards - global_baseline_reward
        
        loss_unclipped = ratio * advantage
        loss_clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
        
        # 直接计算所有样本的平均损失
        policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()
        
        # --- KL 散度惩罚计算 ---
        kl_divergence = log_ratio.mean()
            
        # --- 策略熵计算 ---
        policy_entropy = self._compute_policy_entropy(model_preds, node_masks)

        # --- 总损失 ---
        # 裁剪 KL 散度防止梯度爆炸
        kl_divergence_for_loss = torch.clamp(kl_divergence, min=-10.0, max=10.0) 
        ent_coef = getattr(self.cfg.grpo, 'ent_coef', 0.01)
        total_loss = policy_loss + self.beta * kl_divergence_for_loss - ent_coef * policy_entropy
        
        if global_rank == 0:
            print(f"📊 [Rank 0] GRPO损失: 总计={total_loss.item():.4f}, 策略={policy_loss.item():.4f}, KL={kl_divergence.item():.4f}, 熵={policy_entropy.item():.4f}")

        loss_metrics.update({
            'loss/total_loss': total_loss.item(),
            'loss/policy_loss': policy_loss.item(),
            'loss/kl_divergence': kl_divergence.item(), # 记录原始KL
            'loss/policy_entropy': policy_entropy.item(),
            'grpo_stats/avg_importance_ratio': ratio.mean().item(),
            'grpo_stats/avg_advantage': advantage.mean().item(),
        })

        return {"total_loss": total_loss, "metrics": loss_metrics}
    

    def sample_and_compute_single_group(self, global_rank: int = 0) -> Tuple[List, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        采样并计算单个group，并返回详细的指标字典用于日志记录。
        
        Returns:
            (组数据, 奖励张量, 累积损失张量, 当前对数概率, 参考对数概率, 指标字典)
        """
        start_time = time.time()
        
        # 打印优化: 保留关键的起始信息
        unique_seed = int(time.time() * 1000) % (2**32) + global_rank
        print(f"🎯 [Rank {global_rank}] 采样 group_size={self.group_size}, seed={unique_seed}")
        
        graphs, node_mask, cumulative_loss, current_log_prob, reference_log_prob, model_pred = self.sample_graphs_with_gradients(
            batch_size=self.group_size,
            seed=unique_seed
        )
        
        sampling_duration = time.time() - start_time
        
        graph_list = self._convert_to_graph_list(graphs, node_mask)
        
        reward_start_time = time.time()
        rewards = self.reward_function(graph_list)
        rewards = rewards.to(next(self.model.parameters()).device)
        reward_duration = time.time() - reward_start_time
        
        duration = time.time() - start_time
        
        # 打印优化: 合并为一个清晰的完成摘要
        avg_reward = rewards.mean().item()
        loss_val = cumulative_loss.mean().item()
        print(f"✅ [Rank {global_rank}] Group完成 (耗时 {duration:.2f}s): 平均奖励={avg_reward:.4f}, 累积损失={loss_val:.2f}")

        # SwanLab日志: 创建一个包含所有详细指标的字典
        group_metrics = {
            f"perf/group_duration_sec_rank_{global_rank}": duration,
            f"perf/sampling_duration_sec_rank_{global_rank}": sampling_duration,
            f"perf/reward_duration_sec_rank_{global_rank}": reward_duration,
            f"reward/avg_reward_rank_{global_rank}": avg_reward,
            f"loss/cumulative_loss_rank_{global_rank}": loss_val,
            f"probs/current_log_prob_rank_{global_rank}": current_log_prob.mean().item(),
            f"probs/reference_log_prob_rank_{global_rank}": reference_log_prob.mean().item(),
        }
        
        return [graph_list], rewards, cumulative_loss, current_log_prob, reference_log_prob, model_pred, node_mask, group_metrics


    def state_dict(self) -> Dict:
        """返回包含训练器状态的字典，用于保存检查点"""
        state = {
            'global_step': self.global_step,
        }
        # 仅当 original_model_state 初始化后才保存
        if self.original_model_state is not None:
            state['original_model_state'] = self.original_model_state
        return state

    def load_state_dict(self, state_dict: Dict):
        """从状态字典中加载训练器状态"""
        # self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.global_step = state_dict['global_step']
        # 兼容处理可能不存在的 key
        if 'original_model_state' in state_dict:
            self.original_model_state = state_dict['original_model_state']
        print(f"✅ GRPOTrainer 状态已从检查点加载 (global_step: {self.global_step})")

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        torch.save(self.state_dict(), filepath)
        print(f"检查点已保存到: {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        state_dict = torch.load(filepath, map_location=next(self.model.parameters()).device)
        self.load_state_dict(state_dict)
        # 兼容旧版检查点
        if 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        print(f"检查点已从 {filepath} 加载")