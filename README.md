# Brains in Bits: A Comparative Study of PEFT Techniques for Commonsense Reasoning on GPT-2

While industry convention assumes "bigger is better," training large models fully is resource-intensive, often impractical, and costly to the environment. This project investigates how smaller models like GPT-2 can be selectively enhanced for commonsense reasoning tasks through Parameter-Efficient Fine-Tuning (PEFT) methods.

We compared four PEFT techniques: **LoRA**, **QLoRA**, **Prefix Tuning**, and **IA³**. Our results show that IA³ and Prefix Tuning substantially outperform LoRA and QLoRA, achieving up to **50% validation accuracy** in commonsense categories, with noticeable reductions in perplexity. Each method, applied to the smallest GPT-2 model, outperformed the largest GPT-2 model in zero-shot commonsense reasoning tests.

## 🎯 Key Findings

- **IA³** achieved the highest accuracy at **50%** validation accuracy
- **Prefix Tuning** reached **46%** validation accuracy  
- Both significantly outperformed **LoRA (12%)** and **QLoRA (15%)**
- All PEFT methods on GPT-2 small outperformed zero-shot GPT-2 large (6.35%)
- Perplexity dropped dramatically from >5000 (zero-shot) to ~14 (fine-tuned)

## 📊 Results Summary

| Method | Validation Accuracy | Perplexity |
|--------|-------------------|------------|
| Zero-Shot GPT-2 (1.5B) | 6.35% | 5536.87 |
| LoRA | 12% | 14.31 |
| QLoRA | 15% | 14.45 |
| Prefix Tuning | 46% | 14.06 |
| **IA³** | **50%** | 16.24 |

## 🧠 What are PEFT Methods?

Parameter-Efficient Fine-Tuning (PEFT) methods allow us to adapt large language models by updating only a small subset of parameters, rather than retraining the entire model. This approach:

- **Reduces computational costs** dramatically
- **Minimizes memory requirements**
- **Enables rapid experimentation**
- **Maintains model stability**

### Methods Compared

1. **LoRA (Low-Rank Adaptation)**: Introduces low-rank updates within attention weights
2. **QLoRA**: Combines 4-bit model quantization with LoRA for memory efficiency
3. **Prefix Tuning**: Appends learned prefix tokens to transformer layer inputs
4. **IA³ (Input-Adaptive Attention)**: Adds learnable scaling factors to key, value, and feedforward transformations

## 🗂️ Dataset: CommonsenseQA

We used the CommonsenseQA dataset, which consists of multiple-choice questions testing everyday commonsense knowledge. The dataset includes:

- **785 categories** of commonsense questions
- **Multiple-choice format** with 5 options each
- **Focus on implicit world knowledge** rather than surface patterns
- **Clustered into 10 broad classes** for analysis

### Why CommonsenseQA?

Initially, we attempted mathematical reasoning with OpenMathInstruct-2, but GPT-2's limited capacity for symbolic structures caused training to stall. CommonsenseQA proved a better fit for evaluating PEFT effectiveness on reasoning tasks within GPT-2's capabilities.

## ⚙️ Experimental Setup

### Model Architecture
- **Base Model**: GPT-2 Small (~125M parameters)
- **Hardware**: Single NVIDIA A100 GPU
- **Framework**: HuggingFace PEFT library

### Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Batch Size**: 8
- **Epochs**: 10
- **Evaluation**: Every 100 steps

### Evaluation Metrics
- **Validation Accuracy**: Percentage of correct multiple-choice answers
- **Perplexity**: Model confidence on validation sequences
- **Category-wise Analysis**: Performance across different commonsense domains

## 📈 Detailed Results

### Training Dynamics
All methods showed significant improvements over zero-shot performance, but with different patterns:

- **IA³ and Prefix Tuning**: Rapid convergence with sustained high performance
- **LoRA and QLoRA**: Steady but limited improvement plateaus

### Category-Specific Performance
Different PEFT methods showed varying strengths across commonsense categories:

- **IA³ and Prefix Tuning**: Excelled in abstract reasoning (Emotions, Technology, Buildings & Spaces)
- **LoRA and QLoRA**: More modest, uniform improvements across categories

## 🔍 Key Insights

### Why IA³ and Prefix Tuning Outperformed

We hypothesize that IA³ and Prefix Tuning's superior performance stems from their direct intervention in attention mechanisms, which are central to relational and inferential reasoning. In contrast, LoRA and QLoRA primarily update projection layers, which may be less effective for reshaping internal reasoning processes.

### Qualitative Observations

- **IA³ and Prefix Tuning**: Models selected semantically plausible answers even when incorrect, suggesting partial internalization of reasoning heuristics
- **LoRA and QLoRA**: Models frequently defaulted to seemingly random choices, implying weaker structural learning

## 🚀 Future Work

Several exciting directions for continuation:

1. **Advanced PEFT Techniques**: Explore newer methods and ensemble strategies
2. **Larger Models**: Scale experiments to Mistral, Llama 4, and Qwen 3
3. **Complex Datasets**: Test on more challenging reasoning benchmarks
4. **Ensemble Methods**: Combine different PEFT techniques
5. **Multi-hop Reasoning**: Evaluate on tasks requiring deeper logical chains

## 💡 Implications

This work demonstrates that:

- **Smaller models + PEFT** can compete with larger models on specific tasks
- **Targeted fine-tuning** is more effective than brute-force scaling
- **Environmental impact** of LLM research can be significantly reduced
- **Accessibility** of advanced NLP capabilities can be democratized
<!-- 
## 🛠️ Getting Started

### Prerequisites
```bash
pip install transformers
pip install peft
pip install torch
pip install datasets
```

### Basic Usage
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, IA3Config

# Load base model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Apply IA³ configuration
peft_config = IA3Config(task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)

# Your training loop here...
``` -->

## 📚 References

This work builds upon several key papers in parameter-efficient fine-tuning:

- **LoRA**: Hu et al. (2021) - Low-Rank Adaptation of Large Language Models
- **Prefix Tuning**: Li & Liang (2021) - Prefix-Tuning: Optimizing Continuous Prompts
- **IA³**: Liu et al. (2022) - Few-Shot Parameter-Efficient Fine-Tuning
- **PEFT Survey**: Ding et al. (2023) - Parameter-Efficient Fine-Tuning of Large-Scale Pre-Trained Language Models

## Authors
- **Vishwanath Guruvayur** - University of Virginia - vish@virginia.edu
- **Luke Napolitano** - University of Virginia - ljn5yms@virginia.edu  
- **Doruk Ozar** - University of Virginia - bcp8dm@virginia.edu

For questions about this research, please reach out to any of the authors listed above.

---

*This research was conducted as part of DS-6051 (Decoding Large Language Models) at the University of Virginia.*