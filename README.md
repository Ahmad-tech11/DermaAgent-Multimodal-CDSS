# 🔬 DermaAgent

**Multimodal Vision-Language Agent for Autonomous Skin Lesion Diagnosis & Saliency-Grounded Interpretability**

> A ReAct (Reasoning + Acting) agent that orchestrates deep learning perception tools — MobileNetV2 classification and Grad-CAM visual explainability — to produce trustworthy, interpretable skin lesion diagnostic reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Supported LLM Backends](#supported-llm-backends)
- [Technical Details](#technical-details)

---

## Overview

DermaAgent is a multimodal large model agent system that combines:

1. **MobileNetV2 Classifier** — Deep CNN for 7-class dermoscopic skin lesion classification (HAM10000 dataset)
2. **Grad-CAM Engine** — Custom-implemented Gradient-weighted Class Activation Mapping for visual interpretability
3. **LLM ReAct Agent** — LangChain-based reasoning agent that orchestrates tools and generates structured clinical reports

### Supported Skin Lesion Classes (HAM10000)

| Index | Abbreviation | Full Name | Severity |
|:-----:|:------------:|:----------|:--------:|
| 0 | akiec | Actinic Keratoses | ⚠️ Moderate |
| 1 | bcc | Basal Cell Carcinoma | 🔴 High |
| 2 | bkl | Benign Keratosis-like Lesions | 🟢 Low |
| 3 | df | Dermatofibroma | 🟢 Low |
| 4 | mel | Melanoma | 🔴 Critical |
| 5 | nv | Melanocytic Nevi | 🟢 Low |
| 6 | vasc | Vascular Lesions | 🟢 Low |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Patient Query + Image                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              DermaAgent (ReAct LLM Agent)               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Thought → Action → Observation → ... → Answer    │  │
│  └───────────────────────────────────────────────────┘  │
│         │                           │                    │
│         ▼                           ▼                    │
│  ┌──────────────┐          ┌──────────────────┐         │
│  │  Classifier   │          │  Grad-CAM Tool    │        │
│  │  Tool         │          │  (Explainability) │        │
│  │  (MobileNetV2)│          │                   │        │
│  └──────┬───────┘          └────────┬──────────┘        │
│         │                           │                    │
│         ▼                           ▼                    │
│  Top-3 Diagnosis            Saliency Heatmap            │
│  + Confidence               + Region Analysis           │
│  + Severity                 + Clinical Alignment        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Structured Clinical   │
            │  Diagnostic Report     │
            └────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- pip (Python package manager)

### 2. Install Dependencies

```bash
cd agent_system
pip install -r requirements.txt
```

### 3. Configure LLM Backend

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your preferred LLM API key
# Example for OpenAI:
#   OPENAI_API_KEY=sk-your-key-here
#   LLM_PROVIDER=openai
#   LLM_MODEL=gpt-4o-mini
```

### 4. (Optional) Custom Model Weights

If you have trained MobileNetV2 weights:
```bash
# Place your weights file at:
mkdir -p model
cp /path/to/your/weights.pth model/derma_mobilenetv2.pth

# Or set the environment variable:
# DERMA_MODEL_WEIGHTS=/path/to/weights.pth
```

> **Note**: If no custom weights are provided, the system gracefully falls back to a pretrained ImageNet backbone for immediate testing.

### 5. Run

```bash
# Option A: Gradio Web UI (recommended)
python demo.py

# Option B: CLI mode
python demo.py --cli --image path/to/skin_lesion.jpg
```

---

## Usage

### Gradio Web UI

```bash
python demo.py                     # Default port 7860
python demo.py --port 8080         # Custom port
python demo.py --share             # Public share link
python demo.py --provider openai   # Force OpenAI backend
```

Open `http://localhost:7860` in your browser. Upload a skin lesion image, optionally type a query, and click **Analyze with DermaAgent**.

The UI displays:
- **Original Image** — The uploaded skin lesion
- **Grad-CAM Heatmap** — Saliency overlay showing model attention
- **Agent Reasoning Trace** — Step-by-step Thought/Action/Observation log
- **Diagnostic Report** — Structured clinical assessment

### CLI Mode

```bash
# Basic analysis
python demo.py --cli --image dermoscopy_sample.jpg

# Custom query
python demo.py --cli --image lesion.jpg --query "Is this melanoma?"

# Specify LLM
python demo.py --cli --image lesion.jpg --provider groq --model llama-3.1-70b-versatile
```

### Python API

```python
from agent import run_agent

result = run_agent(
    query="Analyze this suspicious mole.",
    image_path="skin_lesion.jpg",
    provider="openai",
    model_name="gpt-4o-mini",
)

print(result["output"])  # Final diagnostic report
```

---

## Project Structure

```
agent_system/
├── __init__.py                 # Package initialization
├── agent.py                    # ReAct agent with LangChain
├── demo.py                     # CLI + Gradio Web UI demo
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .env                        # Your local configuration (git-ignored)
├── README.md                   # This file
├── TECHNICAL_REPORT.md         # Academic technical report
├── tools/
│   ├── __init__.py             # Tools package init
│   ├── gradcam_engine.py       # Core Grad-CAM implementation (PyTorch)
│   ├── classifier_tool.py      # MobileNetV2 classifier (LangChain tool)
│   └── gradcam_tool.py         # Grad-CAM wrapper (LangChain tool)
├── model/                      # Model weights directory (optional)
│   └── derma_mobilenetv2.pth   # Custom trained weights
├── outputs/                    # Generated heatmaps and results
│   └── cam_*.png               # Grad-CAM overlay images
└── samples/                    # Sample test images (optional)
```

---

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend | `openai`, `qwen`, `groq`, `ollama` |
| `LLM_MODEL` | Model name | `gpt-4o-mini`, `qwen-plus` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `DASHSCOPE_API_KEY` | Alibaba Qwen key | `sk-...` |
| `GROQ_API_KEY` | Groq API key | `gsk_...` |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token | `hf_...` |
| `DERMA_MODEL_WEIGHTS` | Custom model path | `model/weights.pth` |

---

## Supported LLM Backends

| Provider | Models | API Key Env Var |
|----------|--------|-----------------|
| **OpenAI** | GPT-4o, GPT-4o-mini | `OPENAI_API_KEY` |
| **Qwen** | qwen-plus, qwen-turbo | `DASHSCOPE_API_KEY` |
| **Groq** | llama-3.1-70b, mixtral | `GROQ_API_KEY` |
| **HuggingFace** | Mistral-7B, etc. | `HUGGINGFACEHUB_API_TOKEN` |
| **Ollama** | llama3.2, mistral (local) | No key needed |

The system auto-detects available backends from your `.env` configuration. Priority: OpenAI → Qwen → Groq → HuggingFace → Ollama.

---

## Technical Details

### Grad-CAM Implementation

The Grad-CAM engine (`tools/gradcam_engine.py`) is implemented from scratch in PyTorch:

1. **Forward Hook**: Captures activation maps from `model.features[-1]` (MobileNetV2's final convolutional block)
2. **Backward Hook**: Captures gradients flowing back through the target layer
3. **Importance Weights**: Global average pooling of gradients → channel importance weights α_k
4. **Weighted Combination**: Σ_k(α_k · A^k) — weighted sum of activation maps
5. **ReLU**: Retains only positive contributions
6. **Normalization**: Scale to [0, 1] range
7. **Overlay**: Resize to input dimensions, apply JET colormap, blend with original image

### MobileNetV2 Architecture

- **Backbone**: Pretrained on ImageNet (1.4M images, 1000 classes)
- **Classifier Head**: Modified to 7 outputs for HAM10000 skin lesion classes
- **Input**: 224×224 RGB, normalized with ImageNet statistics
- **Inference**: Softmax → Top-K predictions with confidence scores

---

## ⚠️ Disclaimer

This system is designed for **research and educational purposes only**. It is NOT a certified medical device and should NOT be used for actual clinical diagnosis. All AI-generated findings must be validated by qualified healthcare professionals.

---

## License

MIT License — Assessment Module for CAS/University Research Laboratory Evaluation
