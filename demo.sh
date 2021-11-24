CUDA_VISIBLE_DEVICES=0 python train.py --model FaceBagNet --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox
CUDA_VISIBLE_DEVICES=0 python train.py --model ConvMixer --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox
CUDA_VISIBLE_DEVICES=0 python train.py --model MLPMixer --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox
CUDA_VISIBLE_DEVICES=0 python train.py --model VisionPermutator --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox
CUDA_VISIBLE_DEVICES=0 python train.py --model ViT --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox

CUDA_VISIBLE_DEVICES=0 python train_fusion.py --model FaceBagNetFusion --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox
CUDA_VISIBLE_DEVICES=0 python train_fusion.py --model ViTFusion --cycle_num 1 --cycle_inter 2 --save_dir ./model_demox