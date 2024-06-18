env=metaworld_button-press-topdown-v2
trivial_reward=0 # [0, 1, 2, 3] 0: GT reward, 1: zero reward, 2: random reward, 3: negative reward
data_quality=1.0
seed=10


CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=False --config=configs/iql.yaml \
--env=$env --data_quality=$data_quality --trivial_reward=$trivial_reward --seed=$seed
