## 1. Environment Installation

```
1. conda env create -f [environment.yml](https://github.com/NVlabs/FoundationStereo/blob/master/environment.yml)
2. conda run -n foundation_stereo pip install flash-attn
3. conda activate foundation_stereo
```

## 2. Training on UWStereo Dataset

```
python tools/train.py --cfg_file cfgs/foundationstereo/fstereo_uwstereo.yaml
```

## 3. Evaluation

```
python tools/eval.py --cfg_file cfgs/foundationstereo/fstereo_uwstereo.yaml --eval_data_cfg_file cfgs/uwstereo_eval.yaml --pretrained_model your_pretrained_ckpt_path
```

## Our Reproduced Results 

|                         Model                          |         Original Paper |    Ours|     Configuration | 
|:------------------------------------:|:---------------------:|------------------------:|:------------:|
| [FoundationStereo](https://arxiv.org/abs/2501.09898) |     0.33| **0.34**|        [foundationstereo_sceneflow.yaml](../cfgs/foundationstereo/fstereo_sceneflow.yaml)    | 

Access our checkpoint: [BaiduDrive](https://pan.baidu.com/s/1vA6xp9UMGJ3_tUahBrzIcw?pwd=mx7v) or [Google Drive](https://drive.google.com/drive/folders/1f1NrVMHUQqgqBA7Q5Q-pyZB65GNGBkHG?usp=drive_link)
