"""
GRPO (Group Relative Policy Optimization) training script for VERL.
Adapted for CBC-PromptInjection-Defense project.
"""

import sys
import argparse
import json
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from sllmworks import SllmProject
    from sllmworks.train import RLHFTrainArgs
    from sllmworks.utils.logger import get_logger
    HAS_SLLMWORKS = True
except ImportError:
    HAS_SLLMWORKS = False
    print("Warning: sllmworks not available. This script is for reference only.")

import torch

logger = get_logger(__name__) if HAS_SLLMWORKS else print


def main(args):
    """Main training function."""
    stage = args.stage

    # Directory configuration
    OUTPUTS_DIR = args.output_dir
    custom_reward_function = str(Path(__file__).parent / "reward_model.py")
    TRAIN_DATA = args.train_data
    EVAL_DATA = args.eval_data
    MODEL_PATH = args.model_path

    if not HAS_SLLMWORKS:
        print("Configuration reference:")
        print(f"  Output directory: {OUTPUTS_DIR}")
        print(f"  Reward function: {custom_reward_function}")
        print(f"  Train data: {TRAIN_DATA}")
        print(f"  Eval data: {EVAL_DATA}")
        print(f"  Model path: {MODEL_PATH}")
        return

    # Initialize SllmProject
    pp = SllmProject(OUTPUTS_DIR)

    # Cluster configuration
    n_gpus_per_node = args.n_gpus_per_node
    worker_num = args.worker_num
    nnodes = worker_num + 1  # total nodes = 1 master + worker_num

    # Training configuration
    world_size = n_gpus_per_node * nnodes
    micro_batch_size = args.micro_batch_size
    prompt_length = args.prompt_length * 1024
    response_length = args.response_length * 1024
    actor_rollout_ref_rollout_max_num_batched_tokens = args.max_batched_tokens * 1024
    lr = args.learning_rate
    max_step = args.max_steps
    num_samples = args.num_samples  # Number of responses per prompt for GRPO

    # Common engine arguments
    engine_args = {
        "micro_batch_size": micro_batch_size,
        "custom_reward_function_name": "compute_scores",
        "reward_manager": 'batch',
        "custom_reward_function_path": custom_reward_function,
        "trainer_total_epochs": args.epochs,
        "tensor_model_parallel_size": args.tensor_parallel_size,
        "log_prob_micro_batch_size_per_gpu": 1,
        "max_prompt_length": prompt_length,
        "max_response_length": response_length,
        "runtime_command_extend": [
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=256"
        ]
    }

    # VERL-specific arguments
    verl_args = {
        'algorithm_adv_estimator': 'grpo',
        'data_train_batch_size': world_size * micro_batch_size,
        'data_filter_overlong_prompts': True,
        'data_shuffle': True,
        'actor_rollout_ref_actor_optim_lr': lr,
        'actor_rollout_ref_model_use_remove_padding': True,
        'actor_rollout_ref_actor_ppo_mini_batch_size': world_size * micro_batch_size,
        'actor_rollout_ref_actor_ppo_micro_batch_size_per_gpu': micro_batch_size,
        'actor_rollout_ref_actor_use_kl_loss': True,
        'actor_rollout_ref_actor_kl_loss_coef': 0.02,
        'actor_rollout_ref_actor_kl_loss_type': 'low_var_kl',
        'actor_rollout_ref_model_enable_gradient_checkpointing': True,
        'actor_rollout_ref_rollout_log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref_rollout_tensor_model_parallel_size': args.tensor_parallel_size,
        'actor_rollout_ref_rollout_name': 'vllm',
        'actor_rollout_ref_rollout_gpu_memory_utilization': 0.5,
        'actor_rollout_ref_rollout_enable_chunked_prefill': False,
        'actor_rollout_ref_rollout_enforce_eager': False,
        'actor_rollout_ref_rollout_free_cache_engine': False,
        'rollout_n': num_samples,
        'actor_rollout_ref_ref_log_prob_micro_batch_size_per_gpu': 4,
        'actor_rollout_ref_ref_fsdp_config_param_offload': True,
        'algorithm_kl_ctrl_kl_coef': 0.002,
        'trainer_default_local_dir': OUTPUTS_DIR,
        'trainer_critic_warmup': 0,
        'trainer_logger': ['console', 'tensorboard'],
        'trainer_project_name': 'verl_grpo_cbc_defense',
        'trainer_n_gpus_per_node': n_gpus_per_node,
        'trainer_nnodes': nnodes,
        'trainer_test_freq': args.test_freq,
        'trainer_save_freq': args.save_freq,
        'trainer_total_training_steps': max_step,
        'actor_rollout_ref_rollout_max_num_batched_tokens': actor_rollout_ref_rollout_max_num_batched_tokens,
        'val_before_train': False,
        'data_val_batch_size': 4,
        'ppo_mini_batch_size': 32,
    }

    engine_args.update(verl_args)

    # Pre-commands for environment setup
    pre_command = [
        "pip install absl-py nltk immutabledict langdetect",
        "pip install -U pyarrow"
    ]

    # Create RLHF training arguments
    rl_args = RLHFTrainArgs(
        from_model=MODEL_PATH,
        train_dataset=TRAIN_DATA,
        valid_dataset=EVAL_DATA,
        engine_type="verl",
        engine_args=engine_args,
        pre_command=pre_command
    )

    # Remote executor setup (for cluster environment)
    if args.use_remote:
        from sllmworks.executor import ResourceSpec, RemoteExecutor
        resource_spec = ResourceSpec()
        resource_spec.node_num = nnodes
        resource_spec.dima_id = args.dima_id
        resource_spec.memory = args.memory * 1024
        resource_spec.image = args.image
        resource_spec.pcache_store = [
            'share-secspace-ckpt',
            'share-secspace-bizdata',
            'share-secspace-data',
            'share-secspace-package'
        ]
        remote_executor = RemoteExecutor(resource_spec)
        task_result = pp.train_rlhf(
            rl_args,
            remote_executor,
            env_args={"VLLM_USE_V1": "1"}
        )
        print(task_result)
    else:
        # Local training (for reference)
        print("Local training mode - for full training, use remote executor with VERL")
        print("RL args configured successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO Training for CBC Prompt Injection Defense",
        allow_abbrev=False,
        conflict_handler='resolve',
    )

    # Stage and paths
    parser.add_argument('--stage', type=str, default='all',
                        help='Training stage')
    parser.add_argument('--output_dir', type=str,
                        default='./models/grpo_cbc_defense',
                        help='Output directory for trained model')
    parser.add_argument('--train_data', type=str,
                        default='./data/grpo_dataset/train.parquet',
                        help='Path to training data (parquet)')
    parser.add_argument('--eval_data', type=str,
                        default='./data/grpo_dataset/val.parquet',
                        help='Path to evaluation data (parquet)')
    parser.add_argument('--model_path', type=str,
                        default='./models/sft_model',
                        help='Path to initial SFT model')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--micro_batch_size', type=int, default=1,
                        help='Micro batch size per GPU')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of responses per prompt (GRPO n)')

    # Sequence length configuration
    parser.add_argument('--prompt_length', type=int, default=1,
                        help='Max prompt length in K tokens')
    parser.add_argument('--response_length', type=int, default=4,
                        help='Max response length in K tokens')
    parser.add_argument('--max_batched_tokens', type=int, default=8,
                        help='Max batched tokens in K')

    # Parallelism configuration
    parser.add_argument('--n_gpus_per_node', type=int, default=8,
                        help='Number of GPUs per node')
    parser.add_argument('--worker_num', type=int, default=2,
                        help='Number of worker nodes')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Tensor parallel size')

    # Checkpointing
    parser.add_argument('--test_freq', type=int, default=100,
                        help='Evaluation frequency in steps')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Save frequency in steps')

    # Remote execution (optional)
    parser.add_argument('--use_remote', action='store_true',
                        help='Use remote executor')
    parser.add_argument('--dima_id', type=str, default='',
                        help='DIMA resource ID for remote execution')
    parser.add_argument('--memory', type=int, default=1200,
                        help='Memory per node in GB')
    parser.add_argument('--image', type=str, default='',
                        help='Docker image for remote execution')

    args, _ = parser.parse_known_args()
    main(args)
