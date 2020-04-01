# Code for ChaLearn Face Anti-spoofing Attack Detection Challenge @ CVPR2019 by SeuTao
This is the source code for my solution to the [ChaLearn Face Anti-spoofing Attack Detection Challenge](https://competitions.codalab.org/competitions/20853#learn_the_details) hosted by ChaLearn. 
![image](PNG/v1_fusion.png)
## Recent Update

**`2019.3.29`**: Final code is not ready, will update soon.

**`2019.3.10`**: code upload for the origanizers to reproduce.

#### Dependencies
- imgaug==0.2.6
- scikit-image==0.14.0
- scikit-learn==0.19.2
- tqdm==4.23.4
- torch==0.4.1
- torchvision==0.2.1

#### Pretrained models

download [\[models\]](https://drive.google.com/open?id=1YHqPbGOiXlmgHLhc5a9lJrxRS1GIheKJ)

#### Train single-modal Model
train model_A with color imgs， patch size 48：
```
CUDA_VISIBLE_DEVICES=0 python train_CyclicLR.py --model=model_A --image_mode=color --image_size=48
```
infer
```
CUDA_VISIBLE_DEVICES=0 python train_CyclicLR.py --mode=infer_test --model=model_A --image_mode=color --image_size=48
```


#### Train multi-modal fusion model
train model A fusion model with multi-modal imgs， patch size 48：
```
CUDA_VISIBLE_DEVICES=0 python train_Fusion_CyclicLR.py --model=model_A --image_size=48
```
infer
```
CUDA_VISIBLE_DEVICES=0 python train_Fusion_CyclicLR.py --mode=infer_test --model=model_A --image_size=48
```

#### For the origanizers to reproduce final two submissions
unzip the models.zip in the root folder and infer all the trained models 

run ensemble script submission.py to generate the final two submissions in phase2:
(test_first.txt and test_second.txt)
```
python submission.py
```

## Citation
If you find this work or code is helpful in your research, please cite:
```
@InProceedings{Shen_2019_CVPR_Workshops,
author = {Shen, Tao and Huang, Yuyu and Tong, Zhijun},
title = {FaceBagNet: Bag-Of-Local-Features Model for Multi-Modal Face Anti-Spoofing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

## Contact
If you have any questions, feel free to E-mail me via: `taoshen.seu@gmail.com`






