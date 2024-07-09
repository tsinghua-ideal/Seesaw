# Seesaw: Compensating for Nonlinear Reduction with Linear Computations for Private Inference

This is the codebase of [Seesaw: Compensating for Nonlinear Reduction with Linear Computations for Private Inference](https://openreview.net/forum?id=jklD0TV5Hw). 

Seesaw is a novel neural architecture search method tailored for PPML. Seesaw exploits a previously unexplored opportunity to leverage more linear computations and nonlinear result reuse, in order to compensate for the accuracy loss due to nonlinear reduction. It incorporates specifically designed pruning and search strategies, not only to efficiently handle the much larger design space of both linear and nonlinear operators, but also to achieve a better balance between the model accuracy and the online/offline execution latencies. 


## Environment configuration

```
set -e
wget "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
~/miniconda/bin/conda init $(echo $SHELL | awk -F '/' '{print $NF}')
echo 'Successfully installed miniconda...'
echo -n 'Conda version: '
~/miniconda/bin/conda --version
echo -e '\n'
exec bash
```
vim ~/.condarc

```
auto_activate_base: false
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
ssl_verify: false

```

pytorch installation
```
conda create --name nas python=3.8
conda activate nas
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
~~pip install -r requirements.txt~~
while read requirement; do pip3 install $requirement; done < requirements.txt 
```

## Evaluation

### Run

You can run `run_oneshot_cifar100.sh` and `run_oneshot.sh` separately to evalate Seesaw. Remember to replace `data_path` in the shell.
The arguments format is `./run_oneshot_cifar100.sh <model> <function> <Portion of original latency> <batch size>`. 
The format example is: 
```shell
./run_oneshot_cifar100.sh searchcifarsupermodel50 search 0.5 64
./run_oneshot_cifar100.sh searchcifarsupermodel50 retrain 0.5 64

./run_oneshot.sh searchsupermodel50 search 0.5 64
./run_oneshot.sh searchsupermodel50 retrain 0.5 64
```
However, you can still fill these arguments in your shell or run `python main.py` directly like:
```
python main.py --net searchsupermodel50 --dataset imagenet --data_path /home/lifabing/data/ --grad_reg_loss_type add#linear --worker_id 0 --epochs 120 --train_batch_size 64 --ref_latency 0.5 --exported_arch_path ./checkpoints/oneshot/searchsupermodel50/500000/add#linear/checkpoint.json --train_mode search
```

### Out of CUDA memory

Seesaw adopts a large number of linear operators, so if you do not have enough memory to train, replace models.supermodel._SampleLayer.SAMPLE_OPS with:
```
SAMPLE_OPS = [
        'skip_connect',
        'conv_3x3',
        'conv_1x1',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_sep_conv_3x3',
        'conv_3x1_1x3',
        'conv_7x1_1x7',
        'van_conv_3x3'
    ]
```
Note that the accuracy may be lower by 1%-4%.

### Architecture

#### cifar100
cifarsupermodel50 36864
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 0, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 0, "default_16": 1}
```

cifarsupermodel16 45056
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 0, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 1, "default_16": 1}
```

cifarsupermodel50 77824
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 1, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 1, "default_16": 1}
```

cifarsupermodel50 118784
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 1, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 1, "default_9": 1, "default_10": 1, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 0, "default_16": 1}
```

cifarsupermodel101 147456
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 0, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 1, "default_16": 0, "default_17": 0, "default_18": 1, "default_19": 0, "default_20": 1, "default_21": 0, "default_22": 0, "default_23": 1, "default_24": 0, "default_25": 1, "default_26": 0, "default_27": 0, "default_28": 1, "default_29": 1, "default_30": 1, "default_31": 1, "default_32": 1, "default_33": 0}
```

cifarsupermodel50 176128
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 1, "default_5": 1, "default_6": 0, "default_7": 0, "default_8": 1, "default_9": 1, "default_10": 1, "default_11": 1, "default_12": 0, "default_13": 0, "default_14": 1, "default_15": 0, "default_16": 1}


cifarsupermodel80 249856
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 1, "default_5": 0, "default_6": 1, "default_7": 0, "default_8": 1, "default_9": 0, "default_10": 0, "default_11": 1, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 1, "default_16": 0, "default_17": 0, "default_18": 1, "default_19": 1, "default_20": 0, "default_21": 1, "default_22": 1, "default_23": 1, "default_24": 1, "default_25": 1, "default_26": 1}

cifarsupermodel101 307200
```
{"default_1": 1, "default_2": 1, "default_3": 0, "default_4": 1, "default_5": 1, "default_6": 1, "default_7": 1, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 0, "default_12": 0, "default_13": 0, "default_14": 0, "default_15": 0, "default_16": 0, "default_17": 0, "default_18": 0, "default_19": 0, "default_20": 0, "default_21": 0, "default_22": 0, "default_23": 0, "default_24": 0, "default_25": 0, "default_26": 0, "default_27": 0, "default_28": 0, "default_29": 0, "default_30": 0, "default_31": 1, "default_32": 0, "default_33": 1}
```

#### imagenet

supermodel50  112896
```
 {'default_1': 0, 'default_2': 0, 'default_3': 0, 'default_4': 0, 'default_5': 0, 'default_6': 0, 'default_7': 0, 'default_8': 0, 'default_9': 0, 'default_10': 0, 'default_11': 0, 'default_12': 0, 'default_13': 0, 'default_14': 0, 'default_15': 0, 'default_16': 1}
```

supermodel50 363776 
```
{"default_1": 0, "default_2": 0, "default_3": 0, "default_4": 1, "default_5": 0, "default_6": 0, "default_7": 0, "default_8": 0, "default_9": 0, "default_10": 0, "default_11": 1, "default_12": 0, "default_13": 1, "default_14": 1, "default_15": 1, "default_16": 1}
```

supermodel50 564480
```
{'default_1': 1, 'default_2': 0, 'default_3': 0, 'default_4': 1, 'default_5': 0, 'default_6': 0, 'default_7': 0, 'default_8': 0, 'default_9': 0, 'default_10': 0, 'default_11': 1, 'default_12': 0, 'default_13': 1, 'default_14': 1, 'default_15': 1, 'default_16': 1}
```

supermodel50  639744
```
{"default_1": 1, "default_2": 0, "default_3": 0, "default_4": 0, "default_5": 1, "default_6": 1, "default_7": 0, "default_8": 0, "default_9": 1, "default_10": 0, "default_11": 0, "default_12": 1, "default_13": 0, "default_14": 0, "default_15": 1, "default_16": 1}
```

supermodel50 1467648
```
{"default_1": 1, "default_2": 1, "default_3": 1, "default_4": 1, "default_5": 1, "default_6": 1, "default_7": 1, "default_8": 1, "default_9": 1, "default_10": 1, "default_11": 1, "default_12": 1, "default_13": 1, "default_14": 1, "default_15": 1, "default_16": 1}
```


### Train without searching

If you want to reuse the architecture, you can create `checkpoint.json` for ImageNet and `checkpoint2.json` for CIFAR100 in `./checkpoints/oneshot/<model>/<count>/<lossType>`, then set `--spatial` to the shell. After training, just retrain the model without `--spatial` or `--spatial False`.

## Reference

If you use this code, please cite us as follows:
```
@inproceedings{seesaw2024icml,
  title={{Seesaw: Compensating for Nonlinear Reduction with Linear Computations for Private Inference}},
  author={Li, Fabing and Zhai, Yuanhao and Cai, Shuangyu and Gao, Mingyu},
  booktitle={{Proceedings of the International Conference on Machine Learning (ICML 2024)}},
  year={2024}
}

```
