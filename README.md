# 🔬 DermaAgent

**Autonomous Multimodal Vision-Language Architecture for Saliency-Grounded Dermatological Decision Support**

> An interpretable clinical decision support system (CDSS) that couples a deep convolutional neural network (MobileNetV2) with self-implemented Gradient-weighted Class Activation Mapping (Grad-CAM) and a symbolic ReAct (Reasoning + Acting) LLM agent to generate structured, audit-ready Electronic Health Record (EHR) diagnostic reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Demonstration](#system-demonstration)
- [Clinical Taxonomy (8-Class Space)](#clinical-taxonomy-8-class-space)
- [Architecture & Workflow](#architecture--workflow)
- [Empirical Validation & Results](#empirical-validation--results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Grounding](#technical-grounding)
- [Citation & Metadata](#citation--metadata)
- [Disclaimer](#disclaimer)

---

## Overview

Traditional computer-aided diagnosis (CAD) pipelines suffer from **black-box opacity** and lack clinical reasoning capabilities. DermaAgent resolves both bottlenecks through a tripartite multimodal design:

1. **Perception Engine** — a lightweight MobileNetV2 backbone fine-tuned over an 8-class dermatological diagnostic space to yield calibrated prediction probabilities.
2. **Explainability Engine (XAI)** — a self-implemented Grad-CAM engine computing spatial gradient attributions, argmax peak coordinates, anatomical sector categorization, and area coverage ratio ($C_{>0.5}$).
3. **Agentic Orchestration (ReAct)** — a LangChain-powered closed-loop ReAct (Reasoning + Acting) orchestrator enforcing a mandatory two-stage diagnostic protocol before synthesizing standardized clinical assessments.

---

## System Demonstration

### 1. Grad-CAM Attention Alignment

Spatial visual attribution map verifying that deep feature activations correlate with active morphological pathology (scaly, erythematous margins) rather than spurious background artifacts.

<p align="center">
  <img src="assets/figure1_gradcam_overlay.png" alt="Figure 1: Input Dermoscopic Image and Grad-CAM Saliency Overlay" width="700">
  <br>
  <em>Figure 1: Input dermoscopic lesion (left) and Grad-CAM visual feature attribution map (right), with peak localization at (182, 143) px.</em>
</p>

### 2. Autonomous Decision Support Interface 

End-to-end clinical workflow: image acquisition, real-time visual saliency rendering, the agent's step-by-step reasoning trace, and the synthesized EHR-formatted diagnostic report.

<p align="center">
  <img src="assets/figure2_system_ui.png" alt="Figure 2: DermaAgent Gradio Decision Support System" width="850">
  <br>
  <em>Figure 2: End-to-end Gradio decision-support interface displaying the autonomous reasoning trace and synthesized clinical findings.</em>
</p>

---

## Clinical Taxonomy (8-Class Space)

The custom classification head maps deep inverted-residual features ($\mathbb{R}^{1280}$) to an 8-class diagnostic taxonomy:

| Index | Abbr. | Diagnostic Class | Etiological Nature | Typical Severity |
|:-----:|:-----:|:-----------------|:-------------------|:-----------------:|
| 0 | **cp** | Chickenpox | Varicella-zoster viral infection | ⚠️ Moderate |
| 1 | **clm** | Cutaneous Larva Migrans | Parasitic hookworm skin eruption | ⚠️ Moderate |
| 2 | **af** | Athlete Foot (*Tinea pedis*) | Superficial fungal dermatophytosis | ⚠️ Moderate |
| 3 | **imp** | Impetigo | Superficial bacterial pyoderma | ⚠️ Moderate |
| 4 | **nf** | Nail Fungus (*Onychomycosis*) | Subungual fungal invasion | 🟢 Low / Moderate |
| 5 | **cel** | Cellulitis | Deep bacterial dermis/subcutis infection | 🔴 High (Urgent) |
| 6 | **sh** | Shingles (*Herpes zoster*) | Viral reactivation / dermatomal neuropathy | 🔴 High |
| 7 | **rw** | Ringworm (*Tinea corporis*) | Annular fungal lesion | ⚠️ Moderate |

*Severity labels are general clinical triage categories used by the system's design, not per-class empirical test results — only Athlete Foot (index 2) has a validated test run so far (see below).*

---

## Architecture & Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│                    Patient Query + Lesion Image             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               DermaAgent (LangChain ReAct Agent)            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Thought_t → Action_t → Observation_t → ... → Report   │  │
│  └───────────────────────────────────────────────────────┘  │
│               │                             │               │
│               ▼                             ▼               │
│  ┌────────────────────────┐    ┌─────────────────────────┐  │
│  │ skin_lesion_classifier │    │ gradcam_visual_explainer│  │
│  │ (MobileNetV2 Backbone) │    │ (Target Class Gradient) │  │
│  └────────────┬───────────┘    └────────────┬────────────┘  │
│               │                             │               │
│               ▼                             ▼               │
│      Top-3 Ranked Differentials     Spatial Saliency Metric  │
│      + Logit Confidence Margin      Peak (x, y) + Area C>0.5 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│    Structured 6-Section EHR Clinical Diagnostic Assessment   │
│  • Section I: Clinical Indication & Query Formulation        │
│  • Section II: Quantitative Classification & Differentials   │
│  • Section III: Statistical Confidence & Gap Margin Analysis │
│  • Section IV: Saliency Localization & Morphological Match   │
│  • Section V: Clinical Triage & Pharmacological Protocol     │
│  • Section VI: Regulatory & Medicolegal Disclaimer          │
└─────────────────────────────────────────────────────────────┘
```

---

## Empirical Validation & Results

Live end-to-end run on a representative dermoscopic test image (`_uploaded_image.png`):

- **Primary Classification**: Athlete Foot (`af`) — **97.5% confidence**
- **Differential Diagnoses**: Cutaneous Larva Migrans (1.8%), Ringworm (0.3%)
- **Inter-Class Margin**: ΔP = 97.5% − 1.8% = **95.7%**
- **Peak Activation Coordinates**: `(182, 143) px` (center-right anatomical sector), peak intensity 0.6596
- **Area Coverage Ratio ($C_{>0.5}$)**: **10.9%** — categorized as *Highly Focused*

Full methodology, mathematical derivations, and the complete synthesized EHR report are documented in `DermaAgent_Academic_Technical_Report_Muhammad_Ahmad.pdf`.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Ahmad-tech11/DermaAgent-Multimodal-CDSS.git
cd DermaAgent-Multimodal-CDSS

pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
```

Add your LLM inference credentials:

```ini
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Launch System

```bash
# Interactive Gradio web application
python demo.py

# Headless CLI inference
python demo.py --cli --image path/to/lesion.png
```

---

## Project Structure

```text
DermaAgent-Multimodal-CDSS/
├── agent.py # LangChain ReAct agent & prompt engineering
├── demo.py # Dual-mode Gradio interface and CLI harness
├── requirements.txt # Production dependencies
├── .env.example # Environment template
├── README.md # Project documentation
├── assets/ # Architectural diagrams & evaluation screenshots
│   ├── figure1_gradcam_overlay.png
│   └── figure2_system_ui.png
├── tools/
│   ├── init.py # Tool registry
│   ├── classifier_tool.py # Deep perception tool (MobileNetV2)
│   ├── gradcam_engine.py # Standalone Grad-CAM visual attribution engine
│   └── gradcam_tool.py # XAI tool wrapper for ReAct orchestration
└── outputs/
    └── .gitkeep # Target directory for generated saliency maps
```

---

## Technical Grounding

### 1. Saliency Weighting (Grad-CAM)

$$
\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{ij}^k}
$$

$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)
$$

### 2. Saliency Concentration Metric ($C_{>0.5}$)

$$
C_{>0.5} = \frac{1}{N_{\text{total}}} \sum_{u} \sum_{v} \mathbb{I}\left[ L_{\text{norm}}^c(u, v) > 0.5 \right] \times 100\%
$$

- **Highly Focused**: $C_{>0.5} < 15\%$
- **Moderately Focused**: $15\% \le C_{>0.5} < 35\%$
- **Diffuse Attention / Artifact Suspect**: $C_{>0.5} \ge 35\%$

---

## Citation & Metadata

- **Author**: Muhammad Ahmad
- **Affiliation**: Department of Computer Science, COMSATS University Islamabad
- **Repository**: https://github.com/Ahmad-tech11/DermaAgent-Multimodal-CDSS
- **Document Reference**: `DermaAgent_Academic_Technical_Report_Muhammad_Ahmad.pdf`
- **Supported Backends**: Groq Cloud API (`qwen/qwen3.8-27b`)

---

## ⚠️ Disclaimer

DermaAgent is developed strictly for **academic evaluation and decision-support research**. It is not certified as a standalone medical diagnostic device. Final therapeutic interventions must be determined by a qualified dermatologist.
