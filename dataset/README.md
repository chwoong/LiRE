# Dataset

## medium-replay dataset

- MetaWorld medium-replay dataset

- DMControl medium-replay dataset

We used [PEBBLE](https://github.com/rll-research/BPref) repository to create new offline preference-based RL dataset. We collected the replay buffer from `SAC` training using `GT rewards` implemented in [PEBBLE](https://github.com/rll-research/BPref).

You can download dataset [here (Google Drive)](https://drive.google.com/drive/u/2/folders/12jkUR7J3mGCUr9ACsMRda-RobVZO6uGj)

or automatically download:

```
bash ./download.sh
```

`dataset` directory should look like:

```
dataset
├── MetaWorld
│  ├── box-close-v2
|  │  ├── saved_replay_buffer_1000000_seed0
|  │  ├── saved_replay_buffer_1000000_seed1
|  │  ├── saved_replay_buffer_1000000_seed2
│  └── button-press-topdown-v2
│  └── dial-turn-v2
│  └── ...
├── DMControl
│  ├── cheetah-run
│  └── hopper-hop
│  └── humanoid-walk
│  └── ...
```

## medium-expert dataset
For metaworld medium-expert dataset, you can use [IPL](https://github.com/jhejna/inverse-preference-learning) repository

## Human feedback dataset

You can download `human_feedback/metaworld_button-press-topdown-v2/dataset.pkl` dataset [here (Google Drive)](https://drive.google.com/drive/u/2/folders/12jkUR7J3mGCUr9ACsMRda-RobVZO6uGj)



`human_feedback` directory should look like:

```
human_feedback
├── metaworld_button-press-topdown-v2
│  ├── dataset.pkl
│  ├── _Independent.txt
│  ├── _SeqRank.txt
│  ├── _RLT_0.txt
│  ├── _RLT_1.txt
│  ├── render_xxx.gif
│  ├── ...
...

```
