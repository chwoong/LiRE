env=dmc_cheetah-run # env name
data_quality=3.0    # data quality.
                    # The lower the quality, the more random policy data, and the higher the quality, the more expert policy data. (maximum is 10.0)
feedback_num=500    # total feedback number (we use 500, 1000 feedback in the paper)
q_budget=100        # query budget (we use 100 in the paper)
                    # Setting q_budget=1 is equivalent to independent pairwise sampling.
feedback_type=RLT   # ["RLT", "SeqRank"]: RLT means ranked list
model_type=BT       # ["BT", "linear_BT"]: BT means exponential bradley-terry model, and linear_BT use linear score function
epochs=300          # we use 300 epochs in the paper, but more epochs (e.g., 5000) can be used for better performance
activation=tanh     # final activation function of the reward model (use tanh for bounded reward)
seed=10             # random seed
threshold=0.5       # Thresholds for determining tie labels (eqaully preferred pairs)
segment_size=25     # segment size
data_aug=none       # ["none", "temporal"]: if you want to use data augmentation (TDA), set data_aug=temporal
batch_size=512    
ensemble_num=3      # number of reward models to ensemble
ensemble_method=mean    # we average the reward values of the ensemble models
noise=0.0           # probability of preference labels (0.0 is noiseless label and 0.1 is 10% noise label)

# Reward model learning
CUDA_VISIBLE_DEVICES=0 python3 Reward_learning/learn_reward.py --config=configs/reward.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug  --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method --batch_size=$batch_size


# Offline IQL with reward model
CUDA_VISIBLE_DEVICES=0 python3 algorithms/iql.py --use_reward_model=True --config=configs/iql.yaml --env=$env \
--data_quality=$data_quality --feedback_num=$feedback_num --q_budget=$q_budget --feedback_type=$feedback_type --model_type=$model_type \
--threshold=$threshold --activation=$activation --epochs=$epochs --noise=$noise --seed=$seed \
--segment_size=$segment_size --data_aug=$data_aug --ensemble_num=$ensemble_num --ensemble_method=$ensemble_method
