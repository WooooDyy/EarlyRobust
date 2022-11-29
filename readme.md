## EarlyRobust
- This is the implementation for the paper in EMNLP 2022 "Efficient Adversarial Training with Robust Early-Bird Tickets" by Zhiheng Xi, Rui Zheng, Tao Gui, Qi Zhang and Xuanjing Huang. The codes consist of two main parts: Robust Searching stage and drawing & finetuning stage.

    - run searching stage: `sh run_first_stage_imdb.sh`
    - run drawing & fine-tuning stage: `sh run_second_stage_imdb.sh`
## Environments
```
textattack==0.3.4
datasets==1.9.0
pytorch==1.9.0
python==3.7.10
transformers==4.4.2
......
```
- We recommend you to install `transformers==4.4.2` by the following steps:
```shell script
cd transformers-4.4.2
pip install -e .
```

## Citation
todo

