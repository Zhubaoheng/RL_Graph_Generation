"""
GRPO PyTorch Lightning模块
将GRPO训练算法包装到PyTorch Lightning框架中
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Callable, Any
from omegaconf import DictConfig, OmegaConf
import time
import os

try:
    import swanlab
except ImportError:
    swanlab = None

from grpo_trainer import GRPOTrainer
from grpo_rewards import create_reward_function
from graph_discrete_flow_model import GraphDiscreteFlowModel
import utils

class GRPOLightningModule(pl.LightningModule):
    """
    GRPO的PyTorch Lightning实现
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        datamodule,
        model_kwargs,
        total_steps: int,
        **kwargs  # Absorb other potential kwargs
    ):
        super().__init__()
        # 将 `cfg` 中的所有参数提升到 hparams 的顶层
        self.save_hyperparameters(cfg)

        # 将与训练过程相关的参数保存为常规实例属性
        self.cfg = cfg
        self.datamodule = datamodule
        self.model_kwargs = model_kwargs
        self.total_steps = total_steps 

        # 实际模型是一个 GraphDiscreteFlowModel
        self.model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

        # 这些将在 setup() 中被初始化
        self.reward_function = None
        self.grpo_trainer = None
        # 用于在 on_load_checkpoint 和 setup 之间传递状态
        self._grpo_trainer_state_to_restore = None

        print("🚀 GRPO Lightning模块初始化完成 (等待 setup 阶段来创建训练器)")
    
    @staticmethod
    def _get_group_indices_for_rank(num_groups: int, world_size: int, rank: int) -> List[int]:
        """
        计算当前rank需要处理的group索引列表，实现均匀分配。

        例如: 8个group, 3个GPU (world_size=3)
        - rank 0: [0, 1, 2] (3个)
        - rank 1: [3, 4, 5] (3个)
        - rank 2: [6, 7]   (2个)

        Args:
            num_groups: 总group数量。
            world_size: 分布式训练中的总进程数 (GPU数量)。
            rank: 当前进程的rank。

        Returns:
            一个包含该rank应处理的group索引的列表。
        """
        if rank >= num_groups:
            # 如果GPU数量多于group数量，一些GPU将没有任务
            return []

        base_groups_per_gpu = num_groups // world_size
        remainder = num_groups % world_size

        # 前 `remainder` 个GPU会多分配一个group
        if rank < remainder:
            num_groups_for_this_rank = base_groups_per_gpu + 1
            start_index = rank * num_groups_for_this_rank
        else:
            num_groups_for_this_rank = base_groups_per_gpu
            # 计算起始索引时要考虑前面 `remainder` 个GPU多分配的部分
            start_index = remainder * (base_groups_per_gpu + 1) + (rank - remainder) * base_groups_per_gpu
        
        end_index = start_index + num_groups_for_this_rank
        return list(range(start_index, end_index))
    
    def setup(self, stage: str) -> None:
        """在 fit, validate, test, or predict 开始时调用."""
        if stage == "fit":
            print("🔧 在Lightning setup阶段初始化GRPO组件...")
            
            # 1. 创建奖励函数
            # 准备参考指标（如果模型有的话）
            ref_metrics = None
            if hasattr(self.model, 'dataset_info') and hasattr(self.model.dataset_info, 'ref_metrics'):
                ref_metrics = self.model.dataset_info.ref_metrics
            
            self.reward_function = create_reward_function(
                reward_type=self.cfg.grpo.reward_type,
                cfg=self.cfg,
                device=self.device,
                # 传递额外参数以实现向后兼容
                datamodule=self.datamodule,
                model=self.model,
                ref_metrics=ref_metrics,
                name=f"grpo_{self.cfg.grpo.reward_type}"
            )
            
            # 2. 初始化 GRPO 训练器，此时使用未包装的模型
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_function=self.reward_function,
                cfg=self.cfg,
                model_kwargs=self.model_kwargs,
            )
 
            # 3. 如果是从GRPO的checkpoint恢复，则恢复其状态
            if self._grpo_trainer_state_to_restore:
                print("🔄 正在恢复GRPO训练器状态...")
                self.grpo_trainer.load_state_dict(self._grpo_trainer_state_to_restore)
                self._grpo_trainer_state_to_restore = None  # 清理状态

    def on_train_start(self) -> None:
        """
        在训练开始时调用，此时模型已被DDP包装。
        这是更新GRPOTrainer内部模型引用并创建参考模型的最佳时机。
        """
        if self.grpo_trainer:
            # for n,p in self.named_parameters():
            #     if p.requires_grad:
            #         print(n, p.numel())
            # print('总可训练参数数目', sum(p.numel() for p in self.parameters() if p.requires_grad))
            
            print(f"🔧 [on_train_start] 更新GRPOTrainer的模型引用 -> {type(self).__name__}")
            
            # 1. GRPOTrainer.model 应该引用DDP包装后的完整模型
            #    DDP不像FSDP那样分片参数，所以结构更简单
            self.grpo_trainer.model = self

            # 2. GRPOTrainer.core_model 应该引用原始的 GraphDiscreteFlowModel
            #    该实例位于 GRPOLightningModule (self) 上，用于内部算法逻辑。
            self.grpo_trainer.core_model = self.model
            print(f"   -> GRPOTrainer.core_model 已指向: {type(self.grpo_trainer.core_model).__name__}")
            
            # 3. 在此预先创建参考模型，避免在训练步骤中发生冲突
            print("🔧 [on_train_start] 预先创建参考模型...")
            self.grpo_trainer._ensure_reference_model()
            print("✅ [on_train_start] 参考模型已成功创建。")
            
            # 4. 初始化KL惩罚所需的原始模型状态
            #    DDP下每个rank都有完整的参数副本，直接保存即可
            if self.grpo_trainer.original_model_state is None:
                print("🔧 [on_train_start] 初始化KL惩罚的原始模型状态...")
                
                # 直接保存完整的参数状态，DDP确保每个rank都有相同的参数
                params_found = {
                    name: param.clone().detach()
                    for name, param in self.named_parameters()
                    if param.requires_grad
                }
                self.grpo_trainer.original_model_state = params_found
                        # 💡【调试】打印模型中所有可训练参数的名称
  
    def configure_optimizers(self):
        print("🔧 [configure_optimizers] 正在配置优化器和学习率调度器...")
        
        if not hasattr(self, 'grpo_trainer') or self.grpo_trainer is None:
            raise RuntimeError("GRPOTrainer 必须在 configure_optimizers 之前被初始化。")

        target_model = self.grpo_trainer.core_model
        trainable_params = list(target_model.parameters())

        if not trainable_params:
            raise ValueError("[configure_optimizers] 错误: 目标模型中未找到任何参数!")

        print(f"✅ [configure_optimizers] 成功从 GRPOTrainer.core_model 中找到 {len(trainable_params)} 个参数。")

        # 1. 创建优化器
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.hparams.grpo.learning_rate,
            weight_decay=1e-4
        )
        print("✅ 优化器已成功创建。")

        # 2. 创建学习率调度器 (带预热)
        warmup_steps = self.hparams.grpo.get('warmup_steps', 0)
        
        if warmup_steps > 0:
            print(f"🔥 配置学习率预热: {warmup_steps} 步")
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            print("✅ 学习率调度器已成功创建。")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # 每个训练步都更新学习率
                    "frequency": 1,
                },
            }
        else:
            print("✅ 未配置学习率预热。")
            return optimizer

    def configure_gradient_clipping(self, optimizer, optimizer_idx=None, gradient_clip_val=0.0, gradient_clip_algorithm="value"):
        """重写默认梯度裁剪逻辑，当梯度全部为 None 时安全地跳过裁剪。

        兼容DDP策略：解决在首次训练步骤 total_loss 为 0 导致所有参数无梯度时，
        PyTorch 内部 clip_grad_* 调用 _group_tensors_by_device_and_dtype
        报错 `Expected !nested_tensorlist[0].empty() to be true, but got false.` 的问题。
        """

        # 若未设置裁剪或裁剪值为 0，则直接跳过
        if gradient_clip_val is None or gradient_clip_val == 0.0:
            return

        # 检查是否至少存在一个非空梯度（DDP模式下直接检查self的参数）
        has_any_grad = any(p.grad is not None for p in self.parameters())

        if not has_any_grad:
            # 全部梯度为空，直接返回，避免触发内部断言错误
            return

        # 调用 Lightning 提供的 clip_gradients 工具执行实际裁剪
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def training_step(self, batch, batch_idx):
            """
            执行一个训练步骤 - 完整最终版 (已适配熵奖励)
            - 为每个GPU分配多个group进行处理。
            - 收集每个group的详细性能和奖励指标。
            - 在所有GPU间同步数据 (包括熵计算所需数据)，计算全局损失。
            - 只在主进程(Rank 0)上执行日志记录和打印。
            """
            world_size = self.trainer.world_size
            global_rank = self.trainer.global_rank
            num_groups = self.cfg.grpo.num_groups

            # 1. 动态计算当前GPU需要处理的group索引
            group_indices_for_this_rank = self._get_group_indices_for_rank(num_groups, world_size, global_rank)
            
            if not group_indices_for_this_rank:
                # 如果GPU数量多于group数量，某些GPU可能没有任务
                return None

            # 2. 在当前GPU上处理分配到的每一个group，并收集本地数据
            local_rewards_list = []
            local_current_log_probs_list = []
            local_ref_log_probs_list = []
            # 💡 新增: 收集熵计算所需的数据
            local_model_preds_list = []
            local_node_masks_list = []
            local_metrics_to_log = {}

            try:
                # 进入 .eval() 模式进行采样，以获得确定的、可复现的策略评估
                original_mode = self.model.training
                self.model.eval()

                for group_idx in group_indices_for_this_rank:
                    # 💡 修改: 接收所有返回的数据，包括 model_pred 和 node_mask
                    (groups, rewards, cumulative_loss, 
                    current_log_probs, reference_log_probs, 
                    model_pred, node_mask,
                    group_metrics) = self.grpo_trainer.sample_and_compute_single_group(
                        global_rank=global_rank
                    )
                    # 添加到各自的列表中
                    local_rewards_list.append(rewards)
                    local_current_log_probs_list.append(current_log_probs)
                    local_ref_log_probs_list.append(reference_log_probs)
                    # 💡 新增: 收集新数据
                    local_model_preds_list.append(model_pred)
                    local_node_masks_list.append(node_mask)

                    # 更新要记录的指标
                    local_metrics_to_log.update(group_metrics)

            except Exception as e:
                print(f"❌ [GPU {global_rank}] GRPO采样步骤失败: {e}")
                import traceback
                traceback.print_exc()
                return {'loss': torch.tensor(0.0, device=self.device, requires_grad=True)}
            finally:
                # 无论成功与否，都恢复模型原始的训练模式
                self.model.train(original_mode)


            # 3. 在所有GPU间同步数据 (重构以支持自动微分)
            # 3.1. 将本地列表连接成单个张量
            local_rewards = torch.cat(local_rewards_list) if local_rewards_list else torch.empty(0, device=self.device)
            local_current_log_probs = torch.cat(local_current_log_probs_list) if local_current_log_probs_list else torch.empty(0, device=self.device)
            local_ref_log_probs = torch.cat(local_ref_log_probs_list) if local_ref_log_probs_list else torch.empty(0, device=self.device)
            local_node_masks = torch.cat(local_node_masks_list) if local_node_masks_list else torch.empty(0, device=self.device, dtype=torch.bool)

            # 3.2. 特别处理 model_preds, 它们是 PlaceHolder 对象
            if local_model_preds_list and any(p is not None for p in local_model_preds_list):
                local_preds_X = torch.cat([p.X for p in local_model_preds_list])
                local_preds_E = torch.cat([p.E for p in local_model_preds_list])
                valid_y = [p.y for p in local_model_preds_list if p.y is not None]
                local_preds_y = torch.cat(valid_y) if valid_y else torch.empty(0, device=self.device)
            else:
                # 如果当前GPU没有处理任何group，则创建正确形状的空张量
                # 注意: 这里的维度信息需要根据你的模型具体情况调整
                bs, n, c = self.cfg.grpo.group_size, self.cfg.grpo.target_node_count, self.model.output_dims['X']
                local_preds_X = torch.empty((0, n, c), device=self.device)
                local_preds_E = torch.empty((0, n, n, self.model.output_dims['E']), device=self.device)
                local_preds_y = torch.empty((0, self.model.output_dims['y']), device=self.device)

            # 3.3. 使用 all_gather_into_tensor 进行高效、可微分的同步
            def gather_autograd_tensor(tensor: torch.Tensor) -> torch.Tensor:
                """使用 all_gather_into_tensor 同步张量并保留梯度。"""
                if not torch.distributed.is_initialized() or self.trainer.world_size == 1:
                    return tensor
                
                # a. 获取所有GPU上的张量大小 (仅第一个维度)
                local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
                all_sizes = [torch.zeros_like(local_size) for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather(all_sizes, local_size)
                
                # b. 如果所有张量都为空，则无需收集
                if tensor.shape[0] == 0 and all(s.item() == 0 for s in all_sizes):
                    return tensor

                # c. 创建一个足够大的输出张量
                total_size = sum(s.item() for s in all_sizes)
                output_shape = (total_size,) + tensor.shape[1:]
                output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
                
                # d. 执行收集
                torch.distributed.all_gather_into_tensor(output_tensor, tensor)
                return output_tensor

            global_rewards = gather_autograd_tensor(local_rewards)
            global_current_log_probs = gather_autograd_tensor(local_current_log_probs)
            global_ref_log_probs = gather_autograd_tensor(local_ref_log_probs)
            global_node_masks = gather_autograd_tensor(local_node_masks)
            global_preds_X = gather_autograd_tensor(local_preds_X)
            global_preds_E = gather_autograd_tensor(local_preds_E)
            global_preds_y = gather_autograd_tensor(local_preds_y)

            # 3.4. 重新组装 PlaceHolder 对象
            global_model_preds = utils.PlaceHolder(X=global_preds_X, E=global_preds_E, y=global_preds_y)

            # 4. 计算GRPO损失 (现在使用单个全局张量)
            loss_result = self.grpo_trainer.compute_grpo_loss(
                rewards=global_rewards, 
                current_log_probs=global_current_log_probs, 
                reference_log_probs=global_ref_log_probs,
                model_preds=global_model_preds,
                node_masks=global_node_masks,
                global_rank=global_rank
            )
            # 如果损失函数因为数据为空等原因没有返回损失，则我们跳过这个优化步骤
            if "total_loss" not in loss_result:
                return None
                
            loss = loss_result["total_loss"]
            
            # 5. 更新参考模型 (所有进程都需要执行以保持同步)
            # 💡 建议: 将 reference_update_frequency 也放入配置文件中
            update_freq = getattr(self.cfg.grpo, 'ref_model_update_freq', 200)
            self.grpo_trainer._update_reference_model(update_frequency=update_freq)

            # 6. SwanLab日志记录 (只在主进程 Rank 0 上执行)
            if self.trainer.is_global_zero: # 使用 Pytorch Lightning 的推荐方式
                # 6.1 收集并聚合所有GPU上的详细性能指标
                all_gpu_metrics = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_gpu_metrics, local_metrics_to_log)
                
                final_metrics_to_log = {}
                for metrics_dict in all_gpu_metrics:
                    if metrics_dict:
                        final_metrics_to_log.update(metrics_dict)
                
                # 6.2 合并损失计算返回的全局指标
                if "metrics" in loss_result:
                    final_metrics_to_log.update(loss_result["metrics"])
                
                # 6.3 记录学习率
                final_metrics_to_log['learning_rate'] = self.optimizers().param_groups[0]['lr']
                
                # 6.4 一次性记录所有指标
                if swanlab is not None and swanlab.run is not None:
                    swanlab.log(final_metrics_to_log, step=self.grpo_trainer.global_step)
                else:
                    print("Swanlab logger failed.")
            # 7. 更新全局步数 (所有进程都需要知道，以便同步参考模型更新)
            self.grpo_trainer.global_step += 1
            
            return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        保存checkpoint时的回调。添加GRPO特定的状态。
        """
        if self.grpo_trainer:
            # 使用新的 state_dict 方法
            checkpoint["grpo_trainer_state"] = self.grpo_trainer.state_dict()
            global_step = self.grpo_trainer.global_step
            print(f"💾 保存GRPO checkpoint, 全局步数: {global_step}")
        else:
            print("⚠️ 警告: on_save_checkpoint被调用，但grpo_trainer未初始化。")

    def on_train_epoch_end(self):
        """训练周期结束时的回调"""
        # 确保 grpo_trainer 存在
        if hasattr(self, 'grpo_trainer') and self.grpo_trainer:
            current_step = self.grpo_trainer.global_step
            print(f"📊 Epoch {self.current_epoch} 结束, 全局步数: {current_step}")
        else:
            print(f"📊 Epoch {self.current_epoch} 结束 (GRPO训练器未初始化)")
    
    def validation_step(self, batch, batch_idx):
        """验证步骤 - 在GRPO中通常跳过"""
        pass
    
    def get_graph_model(self):
        """获取底层的图生成模型"""
        return self.model
    
    def sample_graphs(self, batch_size: int = None, **kwargs):
        """采样图的便捷方法"""
        if batch_size is None:
            batch_size = self.cfg.grpo.group_size
        
        return self.grpo_trainer.sample_graphs_with_gradients(batch_size=batch_size, **kwargs)
    
    @torch.no_grad()
    def validate_pure_sampling(self, batch_size: int = 8, seed: Optional[int] = 42, save_samples: bool = True):
        """
        纯采样验证：直接调用底层的 GraphDiscreteFlowModel.sample_batch 方法
        并使用原始的 sampling_metrics 进行评估，以验证 checkpoint 的真实质量。
        """
        print("🔍 开始纯采样验证 (调用原始 sample_batch)...")
        self.model.eval()

        # 1. 直接调用原始的、经过验证的采样方法
        print(f"   调用 self.model.sample_batch with batch_size={batch_size}, seed={seed}")
        sampled_graphs, sampled_labels = self.model.sample_batch(
            batch_id=seed,  # 使用种子作为批次ID以确保可复现
            batch_size=batch_size,
            num_nodes=self.cfg.grpo.target_node_count,
            save_final=batch_size if save_samples else 0,
            keep_chain=0,
            number_chain_steps=self.cfg.sample.sample_steps,
            save_visualization=save_samples,
        )
        print(f"   ✅ 成功从原始 sample_batch 生成 {len(sampled_graphs)} 个图")

        # 2. 切换到使用原始的 sampling_metrics 进行评估
        print("   📊 使用原始 sampling_metrics 进行质量评估...")
        
        if not hasattr(self.model, 'sampling_metrics') or self.model.sampling_metrics is None:
            print("   ⚠️ 原始 sampling_metrics 未初始化，跳过评估")
            return {'status': 'sampled_only', 'num_samples': len(sampled_graphs)}

        try:
            # 调用原始的评估函数，它会自己打印详细的MMD分数
            quality_metrics = self.model.sampling_metrics(
                sampled_graphs, # 直接作为位置参数传递
                ref_metrics=self.model.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=0,  # 硬编码，因为没有 trainer
                val_counter=-1,
                test=True, # 标记为测试模式
                local_rank=0, # 硬编码，因为没有 trainer
                labels=sampled_labels if self.model.conditional else None,
            )
            
            # sampling_metrics 对象会自己打印详细日志
            print(f"   📈 原始指标评估完成。请查看上方由 'sampling_metrics' 打印的详细MMD分数。")

            # 兼容之前的返回格式，返回一个包含所有指标的字典
            return quality_metrics

        except Exception as e:
            import traceback
            print(f"   ❌ 原始 sampling_metrics 评估失败: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def forward(self, *args, **kwargs):
        """前向传播 - 委托给图模型"""
        return self.model(*args, **kwargs)


class DummyDataModule(pl.LightningDataModule):
    """
    虚拟数据模块 - 专为GRPO多卡训练设计
    
    GRPO不需要真实的数据加载器，因为图是从模型中采样的。
    这个类提供的数据加载器会生成与num_groups相等的虚拟数据，
    确保DDP能够正确地将不同的group分配给不同的GPU。
    """
    
    def __init__(self, num_groups: int = 1, num_workers: int = 0):
        super().__init__()
        self.num_groups = num_groups
        self.num_workers = num_workers
        print(f"🔧 DummyDataModule初始化: num_groups={num_groups}")
    
    def setup(self, stage: str = None):
        """设置数据"""
        # 创建与num_groups相等数量的虚拟数据，每个数据代表一个group的索引
        self.dummy_data = torch.arange(self.num_groups, dtype=torch.float32).unsqueeze(1)  # [num_groups, 1]
        print(f"🔧 DummyDataModule设置了{len(self.dummy_data)}个虚拟group")
    
    def train_dataloader(self):
        """训练数据加载器 - 每个数据项代表一个group"""
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(self.dummy_data)
        return DataLoader(
            dataset,
            batch_size=self.num_groups,  # 一次性加载所有groups
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return self.train_dataloader()


def create_grpo_lightning_module(
    cfg: DictConfig,
    model_kwargs: dict,
    datamodule,
    total_steps: int,
):
    """
    创建GRPO Lightning模块
    
    Args:
        cfg: 配置对象
        model_kwargs: 模型参数字典
        datamodule: 数据模块
        total_steps: 总训练步数
    """
    print("🔧 正在创建GRPO Lightning模块...")
    
    # 实例化GRPO模块
    grpo_module = GRPOLightningModule(
        cfg=cfg,
        datamodule=datamodule,
        model_kwargs=model_kwargs,
        total_steps=total_steps,
    )
    print("🚀 GRPO Lightning模块实例化完成")
    return grpo_module


def run_grpo_lightning_testing(cfg: DictConfig, checkpoint_path: str):
    """
    使用Lightning框架进行GRPO模型测试/推理
    
    Args:
        cfg: 配置对象
        checkpoint_path: GRPO checkpoint路径
    """
    print(f"🔍 开始GRPO Lightning测试: {checkpoint_path}")
    
    # 设置设备
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    try:
        # 从checkpoint加载GRPO模块
        grpo_module = GRPOLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False,
        )
        grpo_module.eval()
        
        print("✅ GRPO模型加载成功")
        
        # 执行测试逻辑
        with torch.no_grad():
            # 采样一些图进行测试
            test_graphs, _, _, _, _ = grpo_module.sample_graphs(
                batch_size=cfg.grpo.group_size,
            )
            
            print(f"📊 成功采样 {cfg.grpo.group_size} 个测试图")
            
            # 计算奖励
            if hasattr(grpo_module, 'reward_function'):
                graph_list = grpo_module.grpo_trainer._convert_to_graph_list(
                    test_graphs, 
                    torch.ones(cfg.grpo.group_size, test_graphs.X.size(1), dtype=torch.bool, device=device)
                )
                rewards = grpo_module.reward_function(graph_list)
                print(f"🎯 测试图平均奖励: {rewards.mean().item():.4f}")
        
        return grpo_module
        
    except Exception as e:
        print(f"❌ GRPO Lightning测试失败: {e}")
        raise e 