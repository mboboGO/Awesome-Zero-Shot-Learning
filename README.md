# Awesome Zero-Shot Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<p align="center">
  <img src="zsk_diagram.png" width=500>
</p>

A collection of resources on Zero-Shot Learning.

## Why awesome Zero-Shot Learning?
Zero-Shot Learning targets to recognize samples from either seen or unseen classes, which can be applied to image classification, object detection, and semantic segmentation.
This is a collection of papers and resources I curated when learning the ropes in Zero-Shot Learning. I will be continuously updating this list with the latest papers and resources. 
If you want to learn the basics of Zero-Shot Learning and understand how the field has evolved, check out these articles I published on ...


## Contributing

If you think I have missed out on something (or) have any suggestions (papers, implementations and other resources), feel free to [pull a request](https://github.com/mboboGO/awesome-zero-shot-learning/pulls)

Feedback and contributions are welcome!

## Table of Contents
- [Basics](#basics)
- [Papers](#papers)
  - [Emedding-based ZSL](#embedding-based-zsl)
  - [Generative ZSL](#generative-zsl)
  - [ZSL Detection](#zsl-detection)
  - [ZSL Segmentation](#zsl-segmentation)
- [Datasets](#datasets) 
- [Blog posts](#blogposts)
- [Popular implementations](#popular-implementations)
  - [PyTorch](#pytorch)
  - [TensorFlow](#tensorflow)
  - [Torch](#Torch)
  - [Others](#others)

## Basics
- [A 2019 guide to Human Pose Estimation with Deep Learning]


## Papers

### Embedding-based ZSL
- [Hierarchical Disentanglement of Discriminative Latent Features for Zero-shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tong_Hierarchical_Disentanglement_of_Discriminative_Latent_Features_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) - Bin Tong, Chao Wang, Martin Klinkigt, Yoshiyuki Kobayashi, Yuuichi Nonaka. (CVPR 2019)
- [Semantically Aligned Bias Reducing Zero Shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Paul_Semantically_Aligned_Bias_Reducing_Zero_Shot_Learning_CVPR_2019_paper.pdf) - Akanksha Paul, Naraynan C Krishnan, Prateek Munjal. (CVPR 2019)
- [Creativity Inspired Zero-Shot Learning] (https://arxiv.org/pdf/1904.01109.pdf) - Mohamed Elhoseiny, Mohamed Elfeki. (ICCV 2019)
- [Generalized Zero-Shot Recognition based on Visually Semantic Embedding](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Generalized_Zero-Shot_Recognition_Based_on_Visually_Semantic_Embedding_CVPR_2019_paper.pdf) - Pengkai Zhu, Hanxiao Wang, Venkatesh Saligrama. (CVPR 2019)
- [Attentive Region Embedding Network for Zero-shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Attentive_Region_Embedding_Network_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) - Guo-Sen Xie, Li Liu, Xiaobo Jin, Fan Zhu, Zheng Zhang, Jie Qin, Yazhou Yao, Ling Shao. (CVPR 2019)
- [Marginalized Latent Semantic Encoder for Zero-Shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Marginalized_Latent_Semantic_Encoder_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) - Zhengming Ding and Hongfu Liu. (CVPR 2019)
- [Progressive Ensemble Networks for Zero-Shot Recognition](https://arxiv.org/pdf/1805.07473.pdf) - Meng Ye, Yuhong Guo. (Tranductive Learning) (CVPR 2019)
- [Rethinking Knowledge Graph Propagation for Zero-Shot Learning](https://arxiv.org/pdf/1805.11724.pdf) - [[CODE]](https://github.com/cyvius96/adgpm)  - Michael Kampffmeyer, Yinbo Chen, Xiaodan Liang, Hao Wang, Yujia Zhang, and Eric P. Xing. (CVPR 2019)
- [Creativity Inspired Zero-Shot Learning](https://arxiv.org/pdf/1904.01109.pdf) - Mohamed Elhoseiny, Mohamed Elfeki. (ICCV 2019)
- [Generative Dual Adversarial Network for Generalized Zero-shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Generative_Dual_Adversarial_Network_for_Generalized_Zero-Shot_Learning_CVPR_2019_paper.pdf) - [[CODE]](https://github.com/stevehuanghe/GDAN) - He Huang, Changhu Wang, Philip S. Yu Chang-Dong Wang. (CVPR 2019)
- [Rethinking Knowledge Graph Propagation for Zero-Shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kampffmeyer_Rethinking_Knowledge_Graph_Propagation_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) - [[CODE]](https://github.com/cyvius96/DGP) - Michael Kampffmeyer, Yinbo Chen, Xiaodan Liangy, Hao Wang, Yujia Zhang, and Eric P. Xing. (CVPR 2019)
- [Progressive Ensemble Networks for Zero-Shot Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Progressive_Ensemble_Networks_for_Zero-Shot_Recognition_CVPR_2019_paper.pdf) - Meng Ye and Yuhong Guo. (CVPR 2019)
- [Leveraging the Invariant Side of Generative Zero-Shot Learning](https://arxiv.org/pdf/1904.04092.pdf) - Jingjing Li, Mengmeng Jing, Ke Lu, Zhengming Ding, Lei Zhu, Zi Huang. (CVPR 2019)
- [Gradient Matching Generative Networks for Zero-Shot Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sariyildiz_Gradient_Matching_Generative_Networks_for_Zero-Shot_Learning_CVPR_2019_paper.pdf) - Mert Bulent Sariyildiz and Ramazan Gokberk Cinbis. (CVPR 2019)
- [DUAL ADVERSARIAL SEMANTICS-CONSISTENT NETWORK FOR GENERALIZED ZERO-SHOT LEARNING ] - Jian Ni, hanghang Zhang, Haiyong Xie. (NeurIPS 2019)
- [Learning where to look: Semantic-Guided Multi-Attention Localization for Zero-Shot Learning ] - Yizhe Zhu, Jianwen Xie, Zhiqiang Tang, Xi Peng and Ahmed Elgammal. (ICCV 2019)

### ZSL Segmentation
- [Zero-Shot Semantic Segmentation](https://arxiv.org/abs/1906.00817) -- [[CODE]](https://github.com/RohanDoshi2018/ZeroshotSemanticSegmentation) Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, Patrick Pérez. (arXiv19)

### ZSL Detection
- [Zero-Shot Object Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ankan_Bansal_Zero-Shot_Object_Detection_ECCV_2018_paper.pdf) - [[Proj]](http://ankan.umiacs.io/zsd.html) - Ankan Bansal, Karan Sikka, Gaurav Sharma, Rama Chellappa, and Ajay Divakaran. (ECCV 2018)


## Datasets
### Classification
- [CUB]
- [SUN]
- [AWA]
- [AWA2]
- [AWA2]
- [aPY]
- [flowers]
### Segmentation
- [VOC]


## Blog posts


## Popular implementations

### PyTorch


### TensorFlow



### Others

## Todo

- [x] Add basics
- [ ] Add papers on Video Classification
- [ ] Add papers on Video Segmentation
- [ ] Add a SOTA ranking











