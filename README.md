<div align="center">

<!-- <h1>JiuTian (九天) </h1> -->
<h2 class="papername"> <img src="./assets/et.png" style="vertical-align: middle; height: 1em; padding: 0 0.2em;"> CogVLA: Cognition-Aligned Vision-Language-Action Models via Instruction-Driven Routing &amp; Sparsification</h2>
<div>
<div>
    <a href="https://orcid.org/0009-0007-7675-3550" target="_blank">Wei Li</a>,
    <a href="https://scholar.google.com/citations?user=iMJYtvwAAAAJ" target="_blank">Renshan Zhang</a>,
    <a href="https://rshaojimmy.github.io/" target="_blank">Rui Shao*</a>,
    <a href="https://orcid.org/0009-0001-9102-7051" target="_blank">Jie He</a>,
    <a href="https://liqiangnie.github.io/index.html" target="_blank">Liqiang Nie</a>
</div>

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen<br>
*Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2508.21046-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2508.21046)
[![project page](https://img.shields.io/badge/Project-CogVLA-9cf)](https://jiutian-vl.github.io/CogVLA-page/)

<h3 align="center">
  <strong>🛠️ We're still cooking — Stay tuned!🛠️<br>⭐ Give us a star if you like it! ⭐</strong>
</h3>



</div>
</div>



## :fire: Updates

- [08/2025] :fire: [Project page](https://jiutian-vl.github.io/CogVLA-page/) released
- [08/2025] :fire: [arXiv paper](https://arxiv.org/abs/2508.21046) released.

## Introduction

This is the github repository of *CogVLA: Cognition-Aligned Vision-Language-Action Models via Instruction-Driven Routing \& Sparsification*. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 

Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4\% and 70.0\%, respectively, while reducing training costs by 2.5× and decreasing inference latency by 2.8× compared to OpenVLA.

<div align="center">
<img src='assets/introduction.png' width='100%'>
</div>

The overall framework of CogVLA is illustrated below.

<div align="center">
<img src='assets/framework.png' width='100%'>
</div>

## Experiments

**Performance.** CogVLA achieves state-of-the-art performance with success rates of 97.4\% and 70.0\% on simulation and real-world tasks, respectively.

<div align="center">
<img src='assets/main-results2.png' width='100%'>
</div>

**Efficiency.** CogVLA also reduces training costs by 2.5× and decreases inference latency by 2.8× compared to OpenVLA.

<div align="center">
<img src='assets/speed_results.png' width='100%'>
</div>

## Visualization

The attention maps of CogVLA highlight task-relevant regions in the input image, well aligning with human cognition during task execution.

<div align="center">
<img src='assets/attention.png' width='100%'>
</div>

## :fire: Citation

If you find this work useful for your research, please kindly cite our paper.

```
@article{li2025cogvla,
  title={CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing \& Sparsification},
  author={Li, Wei and Zhang, Renshan and Shao, Rui and He, Jie and Nie, Liqiang},
  journal={arXiv preprint arXiv:2508.21046},
  year={2025}
}
```

[![Star History Chart](https://api.star-history.com/svg?repos=JiuTian-VL/CogVLA&Date)](https://www.star-history.com/#JiuTian-VL/CogVLA&Date)
