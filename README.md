## <div align="center"> <i>VideoSAVi</i>: Self-Aligned Video Language Models without Human Supervision </div>

<div align="center">
  <a href="https://videosavi.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=GitHub&color=blue&logo=github"></a> &ensp;
  <a href="https://arxiv.org/abs/2412.00624"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2402.05195&color=B31B1B&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/yogkul2000/VideoSAVi"><img src="https://img.shields.io/static/v1?label=Model Weights&message=HuggingFace&color=yellow"></a> &ensp;
  <br>
</div>

---

## Overview

1. **Novel Self-Training Pipeline**: VideoSAVi introduces a self-training framework that generates synthetic question-answer pairs and preference data without requiring costly human annotations or proprietary APIs, significantly reducing the dependency on manually labeled data.

2. **CLIP-Adjusted Direct Preference Optimization (DPO)**: The model uses a CLIP-based filtering mechanism to align responses with video content, ensuring high-quality and video-grounded outputs for tasks like multi-choice QA, open-ended QA, and temporal reasoning.

3. **State-of-the-Art Performance**: VideoSAVi achieves substantial improvements across multiple video-language benchmarks, outperforming existing models in temporal reasoning (+12%), multi-choice QA (+28%), and zero-shot open-ended QA (+8%).

## Installation
Please install the environment using [LLaVA-NeXT(https://github.com/LLaVA-VL/LLaVA-NeXT)
