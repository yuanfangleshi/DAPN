# DAPN
# 1.Setup

A anaconda envs is recommended:

```
conda create -n DAPN python=3.7.3
conda activate DAPN
pip install numpy==1.16.4
pip install scipy==1.3.0
pip install scikit-learn==0.21.2
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
# 2.Useage

## 2.1 Training

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py --lamba1 1.0 --lamba2 0.15 --lamba3 1.0 --m 0.998 --seed 1111 --epoch 100 --train_n_eposide 100 --n_support 5 --source_data_path ./source_domain/miniImageNet/train --current_data_path ./target_domain/Cars --current_class 49 --pretrain_model_path ./pretrain/399.tar --save_dir checkpoint > record.log 2>&1
```

## 2.2 Testing

```
CUDA_VISIBLE_DEVICES=0  nohup python test.py --n_support 5 --seed 1111 --current_data_path ./target_domain/Cars --current_class 49 --test_n_eposide 600  --model_path ./checkpoint/DAPN_Cars_.tar  >record_t1.log 2>&1 &
```

# 3.Datasets

The datasets used in this paper refer to LDP-net


