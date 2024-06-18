# Listwise Reward Estimation for Offline Preference-based Reinforcement Learning

This is the official implementation of LiRE.

This repository contains offline RL dataset and scripts to reproduce experiments.

The code is based on 
- [CORL library](https://github.com/tinkoff-ai/CORL): Offline Reinforcement Learning library. This library provides single-file implementations of offline RL algorithms.
- [PEBBLE](https://github.com/rll-research/BPref): online Preference-based Reinforcement learning code. We used the SAC implementation of this code to create new offline preference-based RL dataset.


Please visit our [paper](https://openreview.net/attachment?id=If6Q9OYfoJ&name=pdf) and [project page](https://sites.google.com/view/lire-opbrl) for more details.


## Installation


<details>
  <summary>1. Install with conda env file (Click to expand)</summary>
  
  ```
    conda env create -f LiRE.yml
    pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
    pip install git+https://github.com/denisyarats/dmc2gym.git
    pip install gdown
    sudo apt install unzip
  ```
</details>


<details>
  <summary>2. Install with installation list (Click to expand)</summary>
  
  ```
    conda create -n LiRE python=3.9
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install "gym[mujoco_py,classic_control]==0.23.0"
    pip install pyrallis rich tqdm==4.64.0 wandb==0.12.21
    pip install git+https://github.com/denisyarats/dmc2gym.git
    pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
    pip install gdown
    sudo apt install unzip
  ```

  - <details>
    <summary>Trouble shooting (Click to expand)</summary>
    
    - `AttributeError: module 'numpy' has no attribute 'int'` 
      - modify to `dim = int(np.prod(s.shape))` from `dim = np.int(np.prod(s.shape))` in `.../LiRE/lib/python3.9/site-packages/dmc2gym/wrappers.py`
  </details>

</details>



## Algorithms

In this repro, we can run MR, SeqRank, LiRE.

For other baselines, we experimented with the following repo:

| Algorithms     | URL  |
|-------------|-----|
| PT | https://github.com/csmile-1006/PreferenceTransformer |
| DPPO | https://github.com/snu-mllab/DPPO |
| IPL | https://github.com/jhejna/inverse-preference-learning |


## Dataset
For more details, please see [here](./dataset/README.md)
- MetaWorld
- DMControl


## Scripts
Please see [here](./scripts/README.md)

