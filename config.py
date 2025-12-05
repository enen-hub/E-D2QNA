import torch

PARAMS_MULTI_OBJECTIVE = {
    "model": {
        "type": "E-D2QNA",
        "version": "MORL-GA-Cached", 
        "raw_feat_dim": 11 
    },
    "data": {
        "train_instance_path": r"E:\xuexi\xiangmu\chejiandiaodu\shenduQwangluo\E-D2QNA\dataset\la16\la16_K1.3.json"
    },
    "experiment": {
        "seed": 42,
        "name": "ETDQN_MORL"  # 🎓 学术级最终训练
    },
    
    # ===== 目标函数配置 =====
    "objectives_config": {
        "names": ["cmax", "ttotal"],
        "ttotal_mode": "true_tardiness",
        "use_noise": False,
        
        "normalization": {
            "enabled": True,
            "method": "minmax",
            "reference_calibration": True,
            "ema_alpha": 0.08,
            "initial_ideal": [0.0, 0.0],
            "initial_nadir": [800.0, 2500.0]
        }
    },
    
    "scalarization": {
        "mode": "weighted_sum",
        "weights": {"cmax": 0.5, "ttotal": 0.5}
    },
    
    # =================================================================== 
    # 🔥 NSGA-II配置 (锁定"宽而浅"多样性策略) 
    # =================================================================== 
    "nsga2": {
        "population_size": 40,           # ⬇️ 从48降低
        "max_generations": 18,           # ⬆️ 从16提高（补偿种群缩小）
        "crossover_rate": 0.75,          # ⬇️ 从0.78降低
        "mutation_rate": 0.28            # ⬇️ 从0.35降低
    },

    
    
    # ===== 进化训练器配置 =====
    "evolutionary_trainer": {
        "use_normalized_objectives": True,
        "heuristic_topk": 3,
        "heuristic_noise": 0.10,
        "allow_defer": 1,
        "dom_eps": 0.05,
        "archive_size": 150,
        "use_flow_time": True,
        "ref_point": [1.1, 1.1],
        
        "enable_state_cache": True,
        "cache_max_size": 8000,
        
        "early_stopping_generations": 10,
        "elite_preservation": True,
        
        "adaptive_mutation": True,
        "low_candidates_threshold": 4,
        "low_candidates_mutation_rate": 0.42
    },
    
    # =================================================================== 
    # 🔥 Agent配置 (锁定稳定探索策略) 
    # =================================================================== 
    "agent": {
        "epsilon_start": 0.95,
        "epsilon_end": 0.12,             # ✅ 修改：提高探索下限，避免过早收敛
        "epsilon_decay": 0.9985,         # ✅ 修改：放缓探索衰减，维持后期“好奇心”
        "gamma": 0.96,
        "memory_capacity": 100000,       # 🎓 学术标准: 扩展回放缓冲区
        "preference_sampling": "enhanced",
        "extreme_preference_prob": 0.35, # ✅ 锁定成功配置: 平衡极端与中间解
        "use_hybrid_td": True
    },
    
    # =================================================================== 
    # 🚀 训练配置 (核心调整：扩展时长) 
    # =================================================================== 
    "training": {
        "max_iter": 1000,                # ✅ 修改: 第一阶段目标训练至1000轮
        "batch_size": 16,                # ✅ 锁定成功配置
        "learning_rate": 8.5e-6,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gradient_clip_norm": 1.0,
        "use_amp": True,
        
        # 🛡️ 短跑训练保障
        "save_checkpoint_interval": 50,  # ✅ 修改2: 在短跑中密集保存
        "validation_interval": 50,
        "early_stopping_patience": 100   # ✅ 修改3: 适配短跑的早停耐心
    },
    
    # =================================================================== 
    # 🎯 分析/评估与末期微调配置（合并为单一块） 
    # =================================================================== 
    "analysis": {
        # 诊断/评估
        "run_diagnostics": True,
        "save_pareto_history": True,
        "normalized_hypervolume_ref_point": [1.1, 1.1],
        "final_evaluation_rollouts": 300,
        "final_evaluation_temperature": 0.20,
        "final_eval_temp_jitter": 0.08,
        "final_eval_include_random": True,
        "final_eval_random_prob": 0.025,
        "final_hv_ref_point_override": [1.1, 1.1],
        "save_convergence_plots": True,
        "save_pareto_evolution": True,
        "detailed_metrics_logging": True,

        # 精英归档与微调
        "enable_elite_archive": True,
        "elite_capacity": 500,           # ✅ 修改4: 增大精英池容量以存储更多优质解
        "enable_finetune_elite": True,
        "finetune_epochs": 150,
        "elite_ratio_schedule": [0.8, 0.5],
        "finetune_epsilon": 0.03,
        "finetune_lr_mult": 0.5,
        "lambda_bc": 0.15
    }
}
