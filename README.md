<h2 align="center"> <a href="https://arxiv.org/abs/2412.00624">VideoSAVi: Self-Aligned Video Language Models without Human Supervision</a></h2>

<div align="center">

<br>


<a href='https://arxiv.org/abs/2412.00624'><img src='https://img.shields.io/badge/arXiv-2412.00624-b31b1b.svg'></a> &nbsp;
 <a href='https://people-robots.github.io/VideoSAVi/'><img src='https://img.shields.io/badge/Project-Website-blue'></a>&nbsp;
 <a href='https://huggingface.co/datasets/yogkul2000/VideoSAVi'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20VideoSAVi--24K-Dataset-blue'></a>&nbsp;
 <a href='https://huggingface.co/yogkul2000/VideoSAVi'><img src='https://img.shields.io/badge/Model-Weights-yellow'></a>&nbsp;

</div>

## Abstract
Recent advances in video-large language models (Video-LLMs) have led to significant progress in video understanding. Current preference optimization methods often rely on proprietary APIs or ground-truth captions to generate preference data, which is costly and labor-intensive. To address this limitation, we introduce VideoSAVi (Self-Aligned Video Language Model), a self-training pipeline that enables Video-LLMs to reason over video content without external supervision. Our approach includes a self-critiquing mechanism that identifies reasoning errors in the model's initial responses and generates improved alternatives, creating preference pairs directly from video content. VideoSAVi then applies Direct Preference Optimization (DPO) to iteratively train the model, enhancing temporal and spatial reasoning. Experiments show that VideoSAVi achieves state-of-the-art performance on MVBench (74.0%) and delivers significant improvements across other benchmarks, including a 3.9% gain on Perception Test and a 6.8% improvement on EgoSchema compared to baseline models. Our model-agnostic approach is computationally efficient, requiring only 32 frames, offering a promising direction for self-aligned video understanding.

<table class="center">
    <tr>
    <td><img src="assets/pipeline.png" alt="VideoSAVi Overview Diagram"></td>
    </tr>
    <tr>
    <td align="center"><em>Overview of the VideoSAVi self-alignment pipeline.</em></td>
    </tr>
</table>


## 🧰 TODO
- [x] Release Paper.
- [x] Release VideoSAVi fine-tuned model weights (e.g., InternVL2.5 + VideoSAVi).
- [x] Release Inference Code.
- [x] Add detailed evaluation scripts.
- [ ] Release Training Code (Self-alignment pipeline scripts).
- [ ] Release VideoSAVi Preference Dataset.


## 📖 Contents
- [Dataset](#data)
- [Install](#install)
- [Inference](#inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contact](#contact)


## 📝 Data

### Training Data (VideoSAVi Preference Pairs)
* Our framework generates preference pairs $(V, q, r_{dispreferred}, r_{preferred})$ entirely through self-alignment.
* In our experiments, we generated **24,000 preference pairs** (targeting spatial and temporal reasoning) from 4,000 videos sampled from Star, Vidal, and WebVid.
* The process involves:
    1.  Generating reasoning questions ($q$) and initial responses ($r_{dispreferred}$) using the base model.
    2.  Using the model's self-critique mechanism to identify errors in $r_{dispreferred}$.
    3.  Generating a revised, preferred response ($r_{preferred}$) based on the critique.
* The generated preference dataset will be released soon. `[Link to Dataset when available]`

### Evaluation Data
We evaluated VideoSAVi on the following standard benchmarks:
* **General Video Understanding:** MVBench, Perception Test, TempCompass, NeXTQA.
* **Long-Form Video Understanding:** EgoSchema, LongVideoBench.

## 📝 Citation
If you find VideoSAVi useful for your research, please cite our paper:
```bib
@article{kulkarni2024videosavi,
  title={{VideoSAVi}: Self-Aligned Video Language Models without Human Supervision},
  author={Kulkarni, Yogesh and Fazli, Pooyan},
  journal={arXiv preprint arXiv:2412.00624},
  year={2024},
  url={https://arxiv.org/abs/2412.00624}
}
```

## 📪 Contact
For questions about the paper or code, please contact Yogesh Kulkarni (`ykulka10@asu.edu`). You can also open an issue in this GitHub repository for bugs or specific questions related to the code.

