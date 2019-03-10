# Code for ChaLearn Face Anti-spoofing Attack Detection Challenge @ CVPR2019 by SeuTao
This is the source code for my solution to the [ChaLearn Face Anti-spoofing Attack Detection Challenge](https://competitions.codalab.org/competitions/20853#learn_the_details) hosted by ChaLearn. 

## Recent Update

**`2019.3.10`**: code upload for submission.

#### Dependencies
- imgaug==0.2.6
- scikit-image==0.14.0
- scikit-learn==0.19.2
- tqdm==4.23.4
- torch==0.4.1
- torchvision==0.2.1

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













