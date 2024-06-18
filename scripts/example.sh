env=metaworld_button-press-topdown-v2
data_quality=1.0          
feedback_num=500    
epochs=300          
activation=tanh     
seed=10             
threshold=0.5       
segment_size=25     
data_aug=none       
batch_size=512    
ensemble_num=3      
ensemble_method=mean    
noise=0.0           
human=False
# MR
q_budget=1       
feedback_type=RLT   
model_type=BT       
# Reward model learning
CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug  --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--batch_size=$batch_size  --checkpoints_path=logs/

# Offline IQL with reward model
CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--checkpoints_path=checkpoints --max_timesteps=250_000 --eval_freq=5_000


# SeqRank
q_budget=1        
feedback_type=SeqRank   
model_type=BT       
# Reward model learning
CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \ 
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug  --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--batch_size=$batch_size  --checkpoints_path=logs/

# Offline IQL with reward model
CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--checkpoints_path=checkpoints --max_timesteps=250_000 --eval_freq=5_000



# RLT
q_budget=100        
feedback_type=RLT   
model_type=linear_BT       
# Reward model learning
CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env --human=$human \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug  --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--batch_size=$batch_size  --checkpoints_path=logs/

# Offline IQL with reward model
CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method \
--checkpoints_path=checkpoints --max_timesteps=250_000 --eval_freq=5_000
