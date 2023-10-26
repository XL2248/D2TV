# D2TV
Code and Data for EMNLP 2023 Findings paper: D2TV: Dual Knowledge Distillation and Target-oriented Vision Modeling for Many-to-Many Multimodal Summarization

Our code is based on the [CrossSum](https://github.com/csebuetnlp/CrossSum) of huggingface transformers.

# The Dependency:
```
python==3.7.9
pytorch==1.7.1 
torchvision==0.8.2 
torchaudio==0.7.2 
cudatoolkit=10.2
```

# Visual Features Extraction and usage
The visual features extraction code is mainly from [image_feature_extraction](https://github.com/j-min/VL-T5/tree/main/feature_extraction) [1,2]. 

The code incorporating image features is mainly borrowed from [Vg-gplms](https://github.com/hltchkust/vg-gplms).

# Data

All the triplet data <image URLs, source article, source summary, target article, and target summary> used in this work can be downloaded [here](https://drive.google.com/file/d/1fiBICIJtP66WYFUrTIyZLGphbfgyqCLs/view?usp=sharing). We crawled the corresponding images from BBC with the article URLs by [CrossSum](https://github.com/csebuetnlp/CrossSum).

# Traing
For multi-gpu multilingual training (8 gpus), run it like this, take mt5-model for example: 
```
bash seq2seq_img_dualkd_mt5/multimodal_train_m2m_mt5.sh 5 4 11 True 1.0 8.0   # many-to-many setting.
```


# Testing
For testing, run it: 
```
bash seq2seq_img_dualkd_mt5/evaluation_runner.sh model_name
```

# Reference
```
[1] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In ICML, 2021: 1931–1942.
[2] Anderson P, He X, Buehler C, et al. Bottom-up and top-down attention for image captioning and visual question answering[C]. In CVPR. 2018: 6077-6086.
```

# Citation
```
@article{liang2023d,
  title={D $\^{} 2$ TV: Dual Knowledge Distillation and Target-oriented Vision Modeling for Many-to-Many Multimodal Summarization},
  author={Liang, Yunlong and Meng, Fandong and Wang, Jiaan and Xu, Jinan and Chen, Yufeng and Zhou, Jie},
  journal={arXiv preprint arXiv:2305.12767},
  year={2023}
}
```
