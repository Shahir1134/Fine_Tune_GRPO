
# GRPO from Scratch: Understanding Reinforcement Learning for Reasoning Models

**A PyTorch implementation and learning project exploring Group Relative Policy Optimization (GRPO) for mathematical reasoning.**

I'm exploring how reinforcement learning can improve a language model's problem-solving behavior by connecting the mathematics behind GRPO with practical implementation.

The project focuses on understanding the algorithm under the hood: reward functions, group-relative advantage estimation, PPO-style clipped objectives, and KL regularization.

## Table of Contents

- [Overview](#overview)
- [Why GRPO?](#why-grpo)
- [PPO vs. GRPO](#ppo-vs-grpo)
- [Mathematical Formulation](#mathematical-formulation)
- [How GRPO Works](#how-grpo-works)
- [DeepSeek and the History of GRPO](#deepseek-and-the-history-of-grpo)
- [Implementation](#implementation)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Learning Resources](#learning-resources)
- [References](#references)

## Overview

Large language models can generate plausible answers without consistently solving complex reasoning problems correctly. Reinforcement learning offers another way to optimize their behavior: generate candidate responses, evaluate their outcomes, and update the model to increase the probability of desirable responses.

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm introduced in the 2024 DeepSeekMath paper. It builds on ideas from Proximal Policy Optimization (PPO), using the relative rewards of multiple responses to estimate advantages without requiring a separate learned value model.

This project explores GRPO through mathematical derivations and a practical PyTorch training experiment on mathematical reasoning.

## Why GRPO?

Consider the question:

> What is 12 × 13?

Suppose a model generates four candidate responses:

| Response | Final answer | Reward |
|---|---:|---:|
| A | 156 | 1 |
| B | 146 | 0 |
| C | 156 | 1 |
| D | 152 | 0 |

For this illustrative example, the reward function assigns `1` to a correct answer and `0` to an incorrect answer.

GRPO compares the rewards of responses generated for the same question. Responses that perform better than the group average receive positive advantages, while responses that perform worse receive negative advantages.

This creates a learning signal based on relative outcomes rather than requiring a separate critic to estimate the expected value of each state.

## PPO vs. GRPO

### Proximal Policy Optimization (PPO)

PPO was introduced by John Schulman and colleagues in 2017.

Its clipped surrogate objective is:

```math
J_{\mathrm{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\,\hat{A}_t,\ \mathrm{clip}\left( r_t(\theta),\, 1-\epsilon,\, 1+\epsilon \right)\hat{A}_t \right) \right]
```

The probability ratio is:

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
```

Where:

- $\theta$ represents the current policy parameters.
- $\pi_\theta$ is the current policy.
- $\pi_{\theta_{\mathrm{old}}}$ is the policy that generated the sampled action.
- $\hat{A}_t$ is the estimated advantage.
- $\epsilon$ controls the clipping range.

PPO commonly uses a learned value function, or critic, to help estimate advantages. In large language model training, that additional model can add considerable memory and computational costs.

### Group Relative Policy Optimization (GRPO)

GRPO retains PPO-style clipped policy optimization but changes how advantages are estimated.

Instead of training a separate critic, GRPO samples multiple responses to the same question and normalizes their rewards within the group.

| Feature | PPO | GRPO |
|---|---|---|
| Policy optimization | Clipped surrogate objective | PPO-style clipped objective |
| Advantage estimation | Commonly uses a learned critic | Uses group-relative rewards |
| Separate value model | Commonly used | Not required |
| Multiple responses per prompt | Not essential to the algorithm | Central to the algorithm |
| Reference-policy regularization | Depends on the implementation | Included in the original GRPO formulation |

**Key insight:** GRPO is a PPO-derived optimization method. Its distinctive contribution is the group-based advantage estimation mechanism, which removes the need for a separate learned value function in its standard formulation.

## Mathematical Formulation

The following equations summarize the main components of the original GRPO formulation. Specific implementations can differ in their reduction, masking, and normalization conventions.

### 1. Group sampling

For a question $q$, sample a group of $G$ responses from the old policy:

```math
\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid q)
```

Each response receives a reward:

```math
r_1, r_2, \ldots, r_G
```

The reward may be based on exact-answer correctness, a format constraint, or another task-specific criterion.

### 2. Group mean

```math
\bar{r} = \frac{1}{G} \sum_{i=1}^{G} r_i
```

This is the average reward across the sampled responses.

### 3. Group standard deviation

```math
\sigma_r = \sqrt{ \frac{1}{G} \sum_{i=1}^{G} (r_i - \bar{r})^2 }
```

### 4. Normalized advantage

```math
\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \delta}
```

Where:

- $r_i$ is the reward of response $i$.
- $\bar{r}$ is the group mean.
- $\sigma_r$ is the group standard deviation.
- $\delta$ is a small constant for numerical stability.

A positive advantage indicates a response performed better than the group average; a negative advantage indicates it performed worse.

When rewards are identical across all responses, the group has no relative reward variation. Implementations must handle this case safely.

### 5. Token-level probability ratio

For response $i$ at token position $t$:

```math
\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})}
```

This compares the probability of a sampled token under the current policy with its probability under the old policy.

### 6. Clipped policy objective

A simplified token-level GRPO policy objective is:

```math
J_{\mathrm{clip}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}(\theta)\,\hat{A}_i,\ \mathrm{clip}\left( \rho_{i,t}(\theta),\, 1-\epsilon,\, 1+\epsilon \right)\hat{A}_i \right) \right]
```

The clipping operation limits the incentive for excessively large policy probability-ratio changes.

### 7. KL regularization

The original GRPO formulation includes a penalty discouraging the trained policy from drifting too far from a reference policy.

A commonly used estimator is:

```math
D_{\mathrm{KL}}^{\mathrm{est}} = \frac{\pi_{\mathrm{ref}}}{\pi_\theta} - \log\left( \frac{\pi_{\mathrm{ref}}}{\pi_\theta} \right) - 1
```

The probabilities are evaluated for sampled tokens under the relevant conditioning context.

The complete objective is:

```math
J_{\mathrm{GRPO}}(\theta) = J_{\mathrm{clip}}(\theta) - \beta\, \mathbb{E}\left[ D_{\mathrm{KL}}^{\mathrm{est}} \right]
```

Here, $\beta$ controls the strength of reference-policy regularization.

Since this objective is maximized, an implementation using a minimization-based optimizer can minimize its negative:

```math
\mathcal{L}_{\mathrm{GRPO}}(\theta) = -J_{\mathrm{GRPO}}(\theta)
```

The exact loss in a codebase depends on its implementation choices, including token masks, importance-sampling corrections, and loss reduction.

## How GRPO Works

The training loop follows seven conceptual stages:

1. **Sample a question:** Select a mathematical problem from the training dataset.
2. **Generate responses:** Sample multiple candidate solutions from the old policy.
3. **Calculate rewards:** Score the responses using the task's reward function.
4. **Estimate advantages:** Normalize rewards relative to the group.
5. **Calculate the objective:** Apply the clipped policy objective and reference-policy penalty.
6. **Update the model:** Use backpropagation and an optimizer to update trainable parameters.
7. **Evaluate:** Test the trained model on held-out questions to measure generalization.

The key distinction is that GRPO learns from the relative outcomes of generated responses rather than only imitating a fixed set of answers.

## DeepSeek and the History of GRPO

GRPO was introduced by the DeepSeek research team in the paper *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024).

Its motivation included reducing memory requirements compared with conventional PPO setups by eliminating the separate value-function model.

The timeline:

- **2017 — PPO:** John Schulman and colleagues introduced Proximal Policy Optimization.
- **2024 — GRPO:** The DeepSeekMath team introduced Group Relative Policy Optimization.
- **2025 — DeepSeek-R1:** DeepSeek's reasoning-model research brought further attention to reinforcement learning for large-scale reasoning.

GRPO is an important development in reasoning-model training, but it is not the only approach. PPO, DPO, and other optimization methods have different objectives, assumptions, and trade-offs.

## Implementation

This project uses a practical training workflow to explore the GRPO objective and its metrics.

The main components are:

| Component | Responsibility |
|---|---|
| Language model | Generates mathematical responses |
| Tokenizer | Converts text to token IDs |
| Dataset | Provides mathematical questions and reference answers |
| Reward function | Evaluates generated answers |
| Group sampler | Produces multiple responses per question |
| Advantage calculation | Computes group-relative rewards |
| GRPO loss | Implements policy optimization and regularization |
| Optimizer | Updates trainable parameters |
| Evaluation loop | Measures held-out performance |

The experiment includes logging of training loss, mean reward, and average response length, along with plots for monitoring training behavior.

### Running the project

1. Install the dependencies used by the notebook.
2. Configure the model and dataset.
3. Run the training workflow.
4. Export the training metrics.
5. Generate the training plots.
6. Evaluate the base and trained models on the same held-out test set.

See the accompanying notebook for the actual implementation and configuration.

## Evaluation Metrics

Training curves alone do not establish that reasoning improved. The evaluation should distinguish optimization behavior from task performance.

### Training metrics

- **Mean reward:** Tracks the average reward of generated responses.
- **GRPO loss:** Tracks the optimization objective during training.
- **Average response length:** Tracks the number of generated tokens.
- **Training steps:** Shows how metrics evolve over optimization.

### Evaluation metrics

- **Exact-match accuracy:** Fraction of questions whose extracted final answer is correct.
- **Pass@k:** Fraction of questions for which at least one of $k$ independently sampled responses is correct.
- **Inference latency:** Average time required to generate a response.
- **Peak GPU memory:** Measures training or inference memory requirements.

For a meaningful comparison, evaluate the base and trained models on the same held-out questions using consistent answer extraction and decoding settings.

No improvement should be claimed until it has been measured.

## Limitations

GRPO does not automatically guarantee better reasoning.

- **Sparse rewards:** Binary correctness rewards may provide limited learning signals when most responses are incorrect.
- **Group variance:** Small groups can produce noisy advantage estimates.
- **Reward hacking:** A model may exploit weaknesses in the reward function.
- **Reference-policy drift:** Weak regularization can permit undesirable changes.
- **Computational costs:** Generating multiple responses per prompt can be expensive.
- **Evaluation leakage:** Using test questions during training can inflate apparent performance.

Careful reward design, held-out evaluation, and reproducibility are essential.

## Learning Resources

This project is also part of my learning journey through ***Build a Reasoning Model (From Scratch)*** by Sebastian Raschka.

I'm exploring the foundations of reasoning models, connecting the underlying theory to practical implementations in PyTorch. Working on GRPO helps me connect concepts such as reward signals, policy optimization, advantage estimation, and KL regularization with the broader process of training reasoning-capable language models.

My goal is to understand not just how to use reasoning models, but how their training algorithms work under the hood.

**Learning by building. Understanding by implementing.**

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
2. Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).
3. DeepSeek-AI. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).
4. Hugging Face. [TRL GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer).

---

**Project status:** Educational implementation and experimentation. Training configurations and measured evaluation results should be documented from the actual runs.
