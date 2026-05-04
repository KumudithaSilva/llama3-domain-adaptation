# Llama3.2 Domain Adaptation

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama%203.2%203B-7B1FA2" />
  <img src="https://img.shields.io/badge/Technique-QLoRA-1E88E5" />
  <img src="https://img.shields.io/badge/Quantization-4bit-3F51B5" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" />
  <img src="https://img.shields.io/badge/Framework-HuggingFace-27A162" />
  <img src="https://img.shields.io/badge/LLMOps-W%26B-FFBE0B" />
  <img src="https://img.shields.io/badge/Hardware-T4%20GPU%20(15GB)-455A64" />
  <img src="https://img.shields.io/badge/Focus-LLM%20Efficiency%20Research-009688" />
  <img src="https://img.shields.io/badge/License-MIT-45a5d7" />
  
</p>

## Overview

Large Language Models (LLMs) such as Llama 3.2 (3B) are highly capable general-purpose systems, but they often underperform on domain-specific tasks without targeted adaptation. At the same time, traditional full fine-tuning is computationally expensive and resource-intensive, making it impractical in constrained environments.

This repository explores a more efficient alternative: **domain adaptation using Parameter-Efficient Fine-Tuning (PEFT)** specifically **LoRA (Low-Rank Adaptation)** combined with **4-bit quantization (QLoRA)** and **Supervised Fine-Tuning (SFT)**. These techniques significantly reduce memory and compute requirements while maintaining strong performance.

The project is structured as a research-oriented, notebook-based experimentation workflow rather than a production system. It focuses on systematically evaluating how modern LLM adaptation strategies perform under practical hardware constraints, using a **Tesla T4 GPU (15GB VRAM)** alongside approximately **12GB of system RAM**.



## Objectives

This project focuses on the following research questions:

- Evaluate baseline performance of a pretrained Llama 3.2 model  
- Measure how effectively LLMs adapt to a **domain-specific dataset**  
- Analyze **performance vs efficiency trade-offs**  
- Study the impact of **LoRA configurations (rank, alpha, dropout)**  
- Evaluate the effectiveness of **4-bit quantization (QLoRA)**  
- Assess feasibility of **low-resource fine-tuning and inference**  

## Dataset

The experiments use the **`stream_items_prompt_lite` dataset (~22K samples)** which was carefully constructed and curated as part of this project.The dataset was specifically prepared to be compatible with LLM fine-tuning requirements and to support controlled experimentation.

Key characteristics:

- Cleaned and preprocessed  
- Structured in **prompt–completion format** (ideal for LLM fine-tuning)  
- Includes **train, validation, and test splits**  
- Ready-to-use format (no additional preprocessing required)  

This dataset enables **reproducible evaluation** and consistent benchmarking across experiments.


**Example sample:**

```json
{
  "prompt": "How much this stream game cost to the nearest dollar?\n\n\nGame: Rocket Explorer\nPeak CCU: 0\nRequired Age: 0\nDLC Count: 0\nSupports Windows: True\nDescription: Explore and interact with rockets in VR.\n\n\nPrice is $",
  "completion": "12.99"
}
```
## Model & Infrastructure

- **Base Model**: Llama 3.2 (3B parameters)  
- **Quantization**: 4-bit (QLoRA)  
  - Memory reduced from ~6GB → ~2.2GB  
- **Hardware**:
  - Tesla T4 GPU (15GB VRAM)  
  - ~12GB system RAM  


## Fine-Tuning Approach

### 1. Quantization (QLoRA)

- Model loaded in **4-bit precision**  
- Drastically reduces memory footprint  
- Enables training on limited hardware  

### 2. LoRA (Low-Rank Adaptation)

- Base model weights are **frozen**  
- Only a small number of trainable parameters are introduced  
- Reduces training cost while maintaining adaptability  

### 3. Supervised Fine-Tuning (SFT)

- Trains the model using **prompt–completion pairs**  
- Aligns model outputs with domain-specific patterns

### 4. Checkpointing & Hugging Face Integration

- Training checkpoints are periodically saved during fine-tuning
- HuggingFace **HfApi** for efficient dataset access and model checkpoint/version management
- Enables resuming training from the last saved state if interrupted
- Final and intermediate models can be pushed to Hugging Face Hub

### 5. Experiment Tracking (LLMOps)

- Uses Weights & Biases (W&B) for experiment tracking and monitoring
- Logs training metrics such as loss, learning rate, and evaluation scores
- Tracks and compares different configurations and runs
- Provides dashboards for better visibility and reproducibility of experiments

## Training Configuration (Hyperparameters)

These parameters control how the model learns, how much memory it uses, and how efficiently training runs.

### Training Setup

- The model is trained for 1 epoch, meaning it goes through the dataset once. This is typically sufficient for domain adaptation when the goal is to specialize a pretrained model without overfitting. **A batch size of 32 is used to balance training speed and GPU memory usage**.

### Input Handling

- Each input sequence is limited to 170 tokens, **ensuring consistent memory usage and faster processing** while trimming overly long inputs.

### Memory Optimization

- To reduce GPU requirements, the model is loaded using **4-bit quantization**, allowing large models like LLaMA 3.2 to run efficiently on limited hardware.

### Parameter-Efficient Fine-Tuning (LoRA)

- Fine-tuning is performed using LoRA (Low-Rank Adaptation), which **updates only selected parts of the model** instead of all parameters. This makes training more efficient while preserving the pretrained knowledge of the base model.

- A key parameter in LoRA is **alpha, which controls how strongly LoRA updates influence the original model**. Without proper scaling, LoRA updates can become too large and negatively affect the pretrained weights.

- **Rank (r)** is set to 256, **Controls the capacity of the LoRA adaptation**. A higher rank allows the model to learn more task-specific patterns but increases compute and memory usage.

- **Dropout**, it applied to LoRA layers to **reduce overfitting and improve generalization**.


## LoRA Weights and Dimensions Explanation

```python
# Base Model
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
          (k_proj): Linear4bit(in_features=3072, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=3072, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
          (up_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
          (down_proj): Linear4bit(in_features=8192, out_features=3072, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((3072,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=128256, bias=False)
)
```

- The pretrained Llama 3.2 (3B) model consists of **28 Transformer layers**
- Each layer contains:
   - Attention projections: <code>q_proj, k_proj, v_proj, o_proj</code>
   - MLP projections: <code>gate_proj, up_proj, down_proj</code>
- Since LoRA decomposes the update into two low-rank matrices, the weight update is expressed as:

$$
\Delta W = A B
$$

where:

$$
A \in \mathbb{R}^{d_{\text{in}} \times r}, \quad
B \in \mathbb{R}^{r \times d_{\text{out}}}
$$

- Here, \( r \) is the rank, which controls adaptation capacity.

- Instead of training the full weight matrix \( W \), only the low-rank matrices \( A \) and \( B \) are trained, while the original pretrained weights remain frozen.

Example

<code>(k_proj): Linear4bit(in_features=3072, out_features=1024, bias=False)</code>


For this layer:

- $d_{\text{in}} = 3072$
- $d_{\text{out}} = 1024$

### LoRA-style parameterization

$$
\Delta W = A B
$$

$$
A \in \mathbb{R}^{3072 \times r}, \quad
B \in \mathbb{R}^{r \times 1024}
$$

### Number of parameters in original layer

The original weight matrix has:

$$
W \in \mathbb{R}^{3072 \times 1024}
$$

So total parameters:

$$
3072 \times 1024 = 3{,}145{,}728 \text{ parameters}
$$

### LoRA parameters

LoRA adds:

$$
(3072 \times r) + (r \times 1024) = r(3072 + 1024)
$$

$$
= 4096r
$$

Now substitute \( r = 256 \):

$$
4096 \times 256 = 1,048,576
$$

$$
\boxed{1,048,576 \text{ parameters}}
$$

- So instead of ~3.1M parameters, LoRA trains **1,048,576 parameters for this layer**.
- This reduction applies **to every layer where LoRA is applied**.

## Key Observations

- Pretrained LLaMA 3 showed poor performance on the domain-specific task  
- Accuracy was lower than traditional ML models.
- LoRA fine-tuning significantly improved performance in the base model.
- Increasing LoRA rank (32 → 256) gave the most noticeable improvement  
- Training Loss: 0.608 → 0.589 | Validation Loss: 0.628 → 0.596  
- No significant overfitting observed despite higher capacity  
- Applying LoRA to both attention and MLP layers gave only slight improvements  
- Dataset is complex with weak patterns → limits overall performance  
- Performance improvements mainly come from increasing adaptation capacity rather than architectural changes. Given the dataset complexity, the achieved validation loss (~0.596) is a reasonable outcome.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c2d5f3d-4651-4be3-9ca9-b123f1971b9a" width="500" />
  <img src="https://github.com/user-attachments/assets/a787d358-2ee4-4b16-a4b3-c4a4e7fde1b4" width="500" />
</p>

## Future Enhancements

- Explore larger and more diverse domain-specific datasets to improve generalization  
- Experiment with different LoRA configurations (alpha, dropout) for better stability and performance  
- Investigate adaptive or dynamic rank selection instead of fixed LoRA rank  
- Apply better data preprocessing and feature engineering to reduce noise in the dataset  
- Explore ensemble approaches combining LLM outputs with traditional ML models  

## Git Flow Workflow

The project follows a Git Flow–inspired workflow:

- 🌿 main — Stable experiment snapshots  
- 🌱 develop — Active experimentation
- ✨ feature/* — Individual experiments and tuning runs  

---
