"""
使用PyTorch Lightning框架的GRPO训练主文件
与原始训练的main.py保持相同的模式，仅添加GRPO特有的逻辑
"""

import os
import time
import torch
import hydra
import numpy as np
import random
import warnings
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from omegaconf import OmegaConf

try:
    import swanlab
except ImportError:
    swanlab = None

# 导入我们的模型和模块
from graph_discrete_flow_model import GraphDiscreteFlowModel
from grpo_lightning_module import (
    GRPOLightningModule, 
    DummyDataModule,
    create_grpo_lightning_module,
    run_grpo_lightning_testing
)

# 导入我们的Lightning模块
from grpo_lightning_module import (
    GRPOLightningModule, 
    DummyDataModule,
    run_grpo_lightning_testing
)

# 设置警告过滤
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# 设置CUDA优化
torch.set_float32_matmul_precision('medium')  # 处理Tensor Cores警告


def create_datamodule_and_components(cfg: DictConfig):
    """
    创建数据模块和模型组件，与原始main.py保持完全一致的逻辑
    
    Args:
        cfg: 配置对象
        
    Returns:
        tuple: (datamodule, model_kwargs)
    """
    dataset_config = cfg["dataset"]
    
    if dataset_config["name"] in [
        "sbm",
        "comm20", 
        "planar",
        "tree",
    ]:
        from analysis.visualization import NonMolecularVisualization
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
            TreeSamplingMetrics,
        )
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
        
    elif dataset_config["name"] in ["my_tree", "my_planar"]:
        from analysis.visualization import NonMolecularVisualization
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures
        
        if dataset_config["name"] == "my_tree":
            from datasets.my_tree_dataset import MyTreeGraphDataModule, MyTreeDatasetInfos
            from analysis.spectre_utils import TreeSamplingMetrics
            
            datamodule = MyTreeGraphDataModule(cfg)
            dataset_infos = MyTreeDatasetInfos(datamodule, dataset_config)
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:  # my_planar
            from datasets.my_planar_dataset import MyPlanarGraphDataModule, MyPlanarDatasetInfos
            from analysis.spectre_utils import PlanarSamplingMetrics
            
            datamodule = MyPlanarGraphDataModule(cfg)
            dataset_infos = MyPlanarDatasetInfos(datamodule, dataset_config)
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
        
    elif dataset_config["name"] in ["qm9", "guacamol", "moses"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization
        from models.extra_features import ExtraFeatures

        if "qm9" in dataset_config["name"]:
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            dataset_smiles = guacamol_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        else:
            raise ValueError("Dataset not implemented")

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )
        
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }
    
    return datamodule, model_kwargs


def create_grpo_callbacks(cfg: DictConfig):
    """创建GRPO训练的回调函数"""
    callbacks = []
    
    # Checkpoint回调 - 与原始main.py保持一致的方式
    if getattr(cfg.grpo, 'save_model', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/grpo_{cfg.general.name}",
            filename="grpo-{epoch:02d}-{step}",
            save_top_k=-1,  # 保存所有checkpoint
            every_n_train_steps=cfg.grpo.save_every,  # 按步数保存
            save_on_train_epoch_end=False,  # 不在epoch结束时保存
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)
    
    # 学习率监控 - 只有在有logger时才添加
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # callbacks.append(lr_monitor)
    
    return callbacks


def run_grpo_lightning_training(cfg: DictConfig):
    global_rank = int(os.environ.get("RANK", "0"))
    
    if global_rank == 0:
        print("🚀 开始PyTorch Lightning GRPO训练 (主进程 Rank 0)")
        print(f"数据集: {cfg.dataset.name}")
        print(f"奖励函数: {cfg.grpo.reward_type}")
        print(f"预训练模型: {cfg.grpo.pretrained_checkpoint}")
    
    # 🦢 初始化 SwanLab (仅在主进程中进行)
    if global_rank == 0 and swanlab is not None:
        print("🦢 [Rank 0] 正在初始化 SwanLab...")
        try:
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            swanlab.init(
                project=f"GRPO-{cfg.dataset.name}",
                experiment_name=cfg.general.name,
                config=config_dict,
                logdir="./swanlab_logs"
            )
            print("✅ [Rank 0] SwanLab 初始化成功！")
        except Exception as e:
            print(f"⚠️ [Rank 0] SwanLab 初始化失败: {e}")
    
    # 设置随机种子 - 所有进程都需要设置以保证数据加载等部分的同步
    pl.seed_everything(cfg.train.seed)
    if global_rank == 0:
        print(f"🎲 [Rank 0] 使用随机种子: {cfg.train.seed}")

    # GPU设置
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    if use_gpu:
        available_gpus = min(cfg.general.gpus, torch.cuda.device_count())
        print(f"🖥️  可用GPU数量: {available_gpus}")
        for i in range(available_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("🖥️  使用CPU")
        available_gpus = 0
    
    # 1. 创建数据模块和模型组件
    try:
        datamodule, model_kwargs = create_datamodule_and_components(cfg)
        print("✅ 数据模块和模型组件创建成功")
    except Exception as e:
        print(f"❌ 数据模块和模型组件创建失败: {e}")
        raise e
    
    # 2. 检查预训练checkpoint
    if not cfg.grpo.pretrained_checkpoint:
        raise ValueError("❌ 错误: GRPO训练必须指定 pretrained_checkpoint!")
    
    # 3. 创建GRPO Lightning模块
    try:
        # device = torch.device(f"cuda:0" if use_gpu else "cpu") # Trainer会自动处理设备
        
        # 将 trainer 的 total_steps 传递下去
        grpo_module = create_grpo_lightning_module(
            cfg=cfg,
            model_kwargs=model_kwargs,
            datamodule=datamodule,
            total_steps=cfg.grpo.total_steps,
        )
        print("✅ GRPO Lightning模块创建成功")
        
    except Exception as e:
        print(f"❌ GRPO Lightning模块创建失败: {e}")
        raise e
    
    # 4. 创建虚拟数据模块（Lightning框架需要）
    # 重要：num_groups必须等于GPU数量，这样DDP会给每张卡分配一个group
    num_groups = cfg.grpo.num_groups
    if use_gpu and available_gpus != num_groups:
        print(f"⚠️ 警告: GPU数量({available_gpus})与num_groups({num_groups})不匹配")
        print(f"🔧 自动调整: 将num_groups设置为{available_gpus}以匹配GPU数量")
        num_groups = available_gpus
    
    dummy_datamodule = DummyDataModule(num_groups=num_groups, num_workers=0)
    
    # 5. 创建回调函数
    callbacks = create_grpo_callbacks(cfg)
    
    # 6. 创建Lightning Trainer - 与原始main.py保持一致的配置风格
    trainer_kwargs = {
        # 基础配置
        'max_steps': cfg.grpo.total_steps,  # 使用步数而不是epoch
        'max_epochs': -1,  # 无限制epoch，由max_steps控制
        
        # 梯度和优化 - 与原始main.py相似
        'gradient_clip_val': getattr(cfg.grpo, 'gradient_clip_val', 1.0),
        'gradient_clip_algorithm': "norm", # DDP支持norm算法，效果通常更好
        'accumulate_grad_batches': getattr(cfg.grpo, 'gradient_accumulation_steps', 1),
        
        # 验证和保存
        'check_val_every_n_epoch': None,  # 禁用验证
        'val_check_interval': None,
        'num_sanity_val_steps': 0,  # 跳过验证sanity check
        
        # 日志和进度
        'log_every_n_steps': getattr(cfg.grpo, 'log_every_n_steps', 50),
        'enable_progress_bar': True,
        'enable_model_summary': True,
        
        # 回调
        'callbacks': callbacks,
        
        # 调试
        'fast_dev_run': cfg.general.name == "debug",
        
        # 其他
        'deterministic': False,  # GRPO需要随机性
        'benchmark': True,  # 优化CUDA性能
        'logger': [],  # 暂时禁用logger
    }
    
    # GPU配置 - 使用DDP策略实现"先治本，后扩容"方法论
    if use_gpu:
        # 第二阶段：扩容 - 使用标准DDP策略将单卡优化成果线性扩展到多卡
        trainer_kwargs.update({
            'accelerator': "gpu",
            'devices': available_gpus,
            'strategy': DDPStrategy(static_graph=True),  # 明确使用DDP策略并声明静态图
            'precision': '32'
        })
        print(f"✅ 已配置DDP策略（静态图模式）：将在 {available_gpus} 张GPU上进行并行训练")
    else:
        # CPU配置
        trainer_kwargs.update({
            'accelerator': "cpu",
            'devices': 1,
        })
    
    trainer = Trainer(**trainer_kwargs)
    print("✅ Lightning Trainer创建完成")
    

    
    # ------------------- 核心训练逻辑 (重构后) -------------------
    try:
        print("🚀 开始GRPO Lightning训练...")
        print(f"   总步数: {cfg.grpo.total_steps}")
        print(f"   学习率: {cfg.grpo.learning_rate}")
        
        # 检查是恢复训练还是从头开始微调
        ckpt_path_to_resume = cfg.grpo.get('resume_from_checkpoint')

        if ckpt_path_to_resume and os.path.exists(ckpt_path_to_resume):
            # --- 场景 B: 恢复中断的GRPO训练 ---
            print(f"📥 从GRPO checkpoint恢复完整训练状态: {ckpt_path_to_resume}")
            # 直接将ckpt_path传递给fit()，Lightning会处理一切
            trainer.fit(
                model=grpo_module,
                datamodule=dummy_datamodule,
                ckpt_path=ckpt_path_to_resume,
            )
        else:
            # 🔧 DDP分布式环境初始化
            # DDP使用标准的分布式初始化，Lightning会自动处理大部分情况
            if not torch.distributed.is_initialized():
                if "WORLD_SIZE" in os.environ:
                    # 由 torchrun/slurm 等启动器设置的多卡环境
                    rank = int(os.environ["RANK"])
                    world_size = int(os.environ["WORLD_SIZE"])
                    print(f"🌍 检测到启动器环境: Rank {rank}/{world_size}. 初始化分布式组...")
                    # DDP通常使用'nccl'后端进行GPU通信
                    torch.distributed.init_process_group(backend="nccl")
                    print("✅ DDP分布式环境初始化成功 (使用启动器变量)。")
                else:
                    # 单卡运行，手动设置虚拟环境
                    print("🔧 未检测到启动器环境，为单卡训练初始化单进程组...")
                    os.environ["MASTER_ADDR"] = "localhost"
                    os.environ["MASTER_PORT"] = "12355"
                    os.environ["RANK"] = "0"
                    os.environ["WORLD_SIZE"] = "1"
                    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
                    print("✅ 单进程组初始化成功。")

            # --- 场景 A: 从预训练模型开始新的微调 ---
            pretrained_path = cfg.grpo.pretrained_checkpoint
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"📥 从预训练模型加载权重 (开始新的GRPO训练): {pretrained_path}")
                
                # 手动加载权重，但不加载优化器状态
                checkpoint = torch.load(pretrained_path, map_location='cpu')

                # --- 重映射 state_dict 的键 ---
                # 预训练模型是 GraphDiscreteFlowModel，而我们要加载到 GRPOLightningModule 中，
                # GRPOLightningModule 将 GraphDiscreteFlowModel 存在 self.model 属性下。
                # 因此，预训练的键 (e.g., 'model.layers.0.weight') 需要被映射为
                # 'model.model.layers.0.weight' 才能正确加载到DDP包装的模型中。
                remapped_state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    # 为所有来自原始 state_dict 的键添加 'model.' 前缀
                    new_key = f"model.{k}"
                    remapped_state_dict[new_key] = v
                print("✅ 键名重映射完成。")
                
                grpo_module.load_state_dict(remapped_state_dict, strict=False)
                print("✅ 预训练权重已加载到模型。")

          
            trainer.fit(
                model=grpo_module,
                datamodule=dummy_datamodule
            )
        
        print("✅ GRPO Lightning训练完成!")
        
        # 保存最终模型
        final_checkpoint_path = f"/home/ly/max/checkpoints/grpo_{cfg.general.name}/final_model.ckpt"
        trainer.save_checkpoint(final_checkpoint_path)
        print(f"💾 最终模型保存至: {final_checkpoint_path}")
        
    except Exception as e:
        print(f"❌ GRPO Lightning训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # 清理资源
        if use_gpu:
            torch.cuda.empty_cache()
    
    return grpo_module


def run_grpo_lightning_sampling(cfg: DictConfig, checkpoint_path: str):
    """
    使用GRPO模块进行纯采样和评估，使用奖励函数作为指标
    """
    print(f"🚀 开始 GRPO Lightning 采样模式: {checkpoint_path}")
    pl.seed_everything(cfg.train.seed)
    if not torch.distributed.is_initialized():
        if "WORLD_SIZE" in os.environ:
            # 由 torchrun/slurm 等启动器设置的多卡环境
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            print(f"🌍 检测到启动器环境: Rank {rank}/{world_size}. 初始化分布式组...")
            # DDP通常使用'nccl'后端进行GPU通信
            torch.distributed.init_process_group(backend="nccl")
            print("✅ DDP分布式环境初始化成功 (使用启动器变量)。")
        else:
            # 单卡运行，手动设置虚拟环境
            print("🔧 未检测到启动器环境，为单卡采样初始化单进程组...")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            print("✅ 单进程组初始化成功。")
    
    
    # GPU设置
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    
    # 1. 创建数据模块和模型组件 (与训练时相同)
    try:
        datamodule, model_kwargs = create_datamodule_and_components(cfg)
        print("✅ 数据模块和模型组件创建成功")
    except Exception as e:
        print(f"❌ 数据模块和模型组件创建失败: {e}")
        raise e
    
    try:
    # device = torch.device(f"cuda:0" if use_gpu else "cpu") # Trainer会自动处理设备
    # 将 trainer 的 total_steps 传递下去
        grpo_module = create_grpo_lightning_module(
            cfg=cfg,
            model_kwargs=model_kwargs,
            datamodule=datamodule,
            total_steps=cfg.grpo.total_steps,
        )
        print("✅ GRPO Lightning模块创建成功")
        
    except Exception as e:
        print(f"❌ GRPO Lightning模块创建失败: {e}")
        raise e    
    # 2. 使用Lightning的标准方式从checkpoint加载模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # --- 重映射 state_dict 的键 ---
    # 预训练模型是 GraphDiscreteFlowModel，而我们要加载到 GRPOLightningModule 中，
    # GRPOLightningModule 将 GraphDiscreteFlowModel 存在 self.model 属性下。
    # 因此，预训练的键 (e.g., 'model.layers.0.weight') 需要被映射为
    # 'model.model.layers.0.weight' 才能正确加载到DDP包装的模型中。
    print("🔧 正在重映射检查点键名以匹配模型结构...")
    remapped_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        # 为所有来自原始 state_dict 的键添加 'model.' 前缀
        new_key = f"model.{k}"
        remapped_state_dict[new_key] = v
    print("✅ 键名重映射完成。")
    
    grpo_module.load_state_dict(remapped_state_dict, strict=False)
    print("✅ 预训练权重已加载到模型。")
 
     # --- 将模型移动到GPU ---
    if use_gpu:
        print(f"🚀 将模型移动到设备: {device}...")
        grpo_module = grpo_module.to(device)
        print("✅ 模型已成功移动到GPU。")
     # --- GPU移动结束 ---
          
      # 3. 手动初始化GRPO组件 (奖励函数等)
    grpo_module.setup("fit")
      
      # 4. 关键：将模型设置为评估模式
    grpo_module.eval()
    print("✅ 模型已设置为评估模式 (model.eval())")
    
    # 5. 获取采样配置
    num_samples = cfg.grpo.get('num_samples_to_validate', 32)
    batch_size = cfg.grpo.group_size
    print(f"📝 采样配置: 总样本数={num_samples}, 每批次大小={batch_size}")
    
    # 6. 执行采样和评估
    all_rewards = []
    num_batches = 1
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_rewards))
        if current_batch_size <= 0:
            break
            
        print(f"\n--- 正在采样批次 {i+1}/{num_batches}, 数量: {current_batch_size} ---")
        
        # 直接调用 GRPOLightningModule 上的验证方法
        metrics = grpo_module.validate_pure_sampling(
            batch_size=current_batch_size,
            seed=cfg.train.seed + i,  # 为每个批次使用不同的种子
            save_samples=True      # 保存样本以供分析
        )
        
        if 'error' not in metrics:
            print(f"   批次平均奖励: {metrics.get('average_reward', 0):.6f}")
        else:
            print(f"   批次采样失败: {metrics['error']}")

    print("\n" + "="*60)
    print("🎉 采样和评估完成！")
    print("="*60)


@hydra.main(version_base="1.3", config_path="../configs", config_name="grpo_lightning_config")
def main(cfg: DictConfig):
    """
    GRPO Lightning训练的主入口函数
    与原始main.py保持相同的结构，支持训练和测试模式
    """
    try:
        # 决定运行模式
        if cfg.grpo.get('sample_only', None):
            # 新增的纯采样模式
            checkpoint_path = cfg.grpo.sample_only
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"采样checkpoint不存在: {checkpoint_path}")
            
            run_grpo_lightning_sampling(cfg, checkpoint_path)
            
        elif getattr(cfg.grpo, 'test_only', None):
            # 测试模式 (保留旧的test_only逻辑)
            checkpoint_path = cfg.grpo.test_only
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"测试checkpoint不存在: {checkpoint_path}")
            
            run_grpo_lightning_testing(cfg, checkpoint_path)
            
        else:
            # 训练模式
            grpo_module = run_grpo_lightning_training(cfg)
        
        print("🎉 GRPO Lightning执行完成!")
        
    except Exception as e:
        print(f"❌ GRPO Lightning执行失败: {e}")
        raise e


if __name__ == "__main__":
    main() 