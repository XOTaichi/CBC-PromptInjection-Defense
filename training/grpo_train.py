"""
GRPO (Group Relative Policy Optimization) training script using VERL.
Uses VERL's command-line interface for training.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_verl_config(args) -> dict:
    """
    Generate VERL configuration dictionary.

    Returns:
        Dictionary containing all VERL configuration parameters
    """
    # Calculate world size
    world_size = args.n_gpus_per_node * (args.worker_num + 1)

    config = {
        # Algorithm settings
        'algorithm.adv_estimator': 'grpo',
        'rollout.n': args.num_samples,

        # Data settings
        'data.train_batch_size': world_size * args.micro_batch_size,
        'data.filter_overlong_prompts': True,
        'data.shuffle': True,
        'data.val_batch_size': 4,

        # Actor/Policy settings
        'actor_rollout_ref.actor.optim.lr': args.learning_rate,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.actor.ppo.mini_batch_size': world_size * args.micro_batch_size,
        'actor_rollout_ref.actor.ppo.micro_batch_size_per_gpu': args.micro_batch_size,
        'actor_rollout_ref.actor.use_kl_loss': True,
        'actor_rollout_ref.actor.kl_loss_coef': 0.02,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size': args.tensor_parallel_size,

        # Rollout settings
        'actor_rollout_ref.rollout.log_prob.micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': args.tensor_parallel_size,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.5,
        'actor_rollout_ref.rollout.enable_chunked_prefill': False,
        'actor_rollout_ref.rollout.enforce_eager': False,
        'actor_rollout_ref.rollout.free_cache_engine': False,
        'actor_rollout_ref.rollout.max_num_batched_tokens': args.max_batched_tokens * 1024,

        # Reference model settings
        'actor_rollout_ref.ref.log_prob.micro_batch_size_per_gpu': 4,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,

        # KL controller
        'algorithm.kl_ctrl.kl_coef': 0.002,

        # Trainer settings
        'trainer.default_local_dir': args.output_dir,
        'trainer.critic_warmup': 0,
        'trainer.logger': ['console', 'tensorboard'],
        'trainer.project_name': 'verl_grpo_cbc_defense',
        'trainer.n_gpus_per_node': args.n_gpus_per_node,
        'trainer.nnodes': args.worker_num + 1,
        'trainer.test_freq': args.test_freq,
        'trainer.save_freq': args.save_freq,
        'trainer.total_training_steps': args.max_steps,
        'trainer.total_epochs': args.epochs,
        'val_before_train': False,
        'ppo_mini_batch_size': 32,
    }

    return config


def build_verl_command(args) -> list:
    """
    Build VERL command line arguments.

    Returns:
        List of command line arguments for verl
    """
    cmd = [
        'verl',
        'train',
        '--config', 'algorithm=grpo',
        f'--config.data.train_path={args.train_data}',
        f'--config.data.val_path={args.eval_data}',
        f'--config.model.path={args.model_path}',
        f'--config.trainer.default_local_dir={args.output_dir}',
        f'--config.rollout.n={args.num_samples}',
        f'--config.actor_rollout_ref.actor.optim.lr={args.learning_rate}',
        f'--config.trainer.total_training_steps={args.max_steps}',
        f'--config.trainer.total_epochs={args.epochs}',
        f'--config.trainer.test_freq={args.test_freq}',
        f'--config.trainer.save_freq={args.save_freq}',
        f'--config.trainer.n_gpus_per_node={args.n_gpus_per_node}',
        f'--config.trainer.nnodes={args.worker_num + 1}',
        f'--config.actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_parallel_size}',
        f'--config.actor_rollout_ref.actor.ulysses_sequence_parallel_size={args.tensor_parallel_size}',
        f'--config.actor_rollout_ref.rollout.max_num_batched_tokens={args.max_batched_tokens * 1024}',
    ]

    # Custom reward function
    if args.reward_function:
        cmd.extend([
            f'--config.reward_manager=batch',
            f'--config.custom_reward_function_name=compute_scores',
            f'--config.custom_reward_function_path={args.reward_function}',
        ])

    return cmd


def main():
    """Main function to run VERL training."""
    parser = argparse.ArgumentParser(
        description="GRPO Training using VERL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to initial SFT model')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (parquet)')
    parser.add_argument('--eval_data', type=str, required=True,
                        help='Path to evaluation data (parquet)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained model')

    # Reward function
    parser.add_argument('--reward_function', type=str,
                        default=str(Path(__file__).parent / 'reward_model.py'),
                        help='Path to custom reward function')

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

    # Mode
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without running')

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("VERL GRPO Training Configuration")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Train data: {args.train_data}")
    print(f"Eval data: {args.eval_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Reward function: {args.reward_function}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)

    # Build command
    cmd = build_verl_command(args)
    cmd_str = ' '.join(cmd)

    print("\nVERL Command:")
    print(cmd_str)
    print()

    if args.dry_run:
        print("Dry run - not executing command")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run VERL command
    print("Starting VERL training...")
    try:
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent))
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'verl' command not found. Please install VERL first.")
        print("\nAlternative: Use the configuration above with verl command directly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
