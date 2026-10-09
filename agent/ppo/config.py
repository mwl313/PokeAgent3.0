"""PPO hyperparameters from the Full Spec 1.1 §8.3 table and ``configs/train.yaml``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 3.0e-4  # peak LR
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-5
    weight_decay: float = 0.0

    # Epochs / batching
    ppo_epochs: int = 4
    global_minibatch_size: int = 4096
    microbatch_size: int = 256
    per_rank_minibatch_size: int = 2048
    grad_accumulation_per_rank: int = 8
    drop_last_minibatch: bool = False
    pad_and_mask_final_minibatch: bool = True
    sample_weighted_ddp_reduction: bool = True
    # Exact full-minibatch objective under gradient accumulation: every
    # microbatch contributes its share of the minibatch's valid actor rows (for
    # policy/entropy/KL) and valid value rows (for the value term) instead of
    # the legacy sample_weight/microbatch-count approximation.
    exact_row_weighted_accumulation: bool = True

    # PPO objective
    clip_epsilon: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    value_loss: str = "half_mse"
    value_clip: bool = False
    max_grad_norm: float = 0.5
    advantage_normalization: str = "iteration_actor_rows"
    advantage_std_floor: float = 1.0e-8
    entropy_coefficient: float = 0.01
    uniform_kl_coefficient: float = 0.001
    target_approx_kl: float = 0.03
    kl_response: str = "end_remaining_epochs_only_then_new_rollout"
    action_temperature: float = 1.0

    # Learning-rate schedule (clocked by committed natural training matches)
    lr_clock: str = "committed_natural_training_matches"
    warmup_matches: int = 250_000
    warmup_start_lr: float = 1.0e-5
    peak_lr: float = 3.0e-4
    final_lr: float = 3.0e-5
    horizon_matches: int = 100_000_000

    # Precision
    mixed_precision: str = "fp16"
    probability_dtype: str = "float32"
    amp_grad_scaler: bool = True
    fp16_autocast: bool = True

    seed: int = 20261006

    @property
    def effective_accumulation(self) -> int:
        """Microbatches per optimizer step that preserve the global batch."""
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        return max(1, self.global_minibatch_size // self.microbatch_size)

    def validate(self) -> None:
        if self.optimizer != "adam":
            raise ValueError("the spec pins the Adam optimizer")
        if self.learning_rate != self.peak_lr:
            raise ValueError("peak learning rate must be 3e-4")
        if self.weight_decay != 0.0:
            raise ValueError("weight decay is pinned to 0")
        if self.value_clip:
            raise ValueError("value clipping is not used")
        if self.value_loss != "half_mse":
            raise ValueError("value loss is 0.5 * MSE")
        if self.microbatch_size > self.global_minibatch_size:
            raise ValueError("microbatch cannot exceed the global minibatch")
        if self.global_minibatch_size % self.microbatch_size != 0:
            raise ValueError(
                "gradient accumulation must preserve the effective global batch"
            )
        if self.action_temperature != 1.0:
            raise ValueError("the spec pins the sampling temperature to 1.0")
        if not 0.0 < self.target_approx_kl:
            raise ValueError("target approximate KL must be positive")
