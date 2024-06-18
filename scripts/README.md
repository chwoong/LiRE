# Dataset

The `data_quality` values for each dataset are as follows:

<details>
  <summary>Data quality (Click to expand)</summary>
  
  | MetaWorld Dataset     | data_qualtiy  |
  |-------------|-----|
  |`button-press-topdown-v2`| 1.0|
  |`box-close-v2`| 8.0|
  |`dial-turn-v2`| 3.0|
  |`sweep-v2`| 7.0|
  |`button-press-topdown-wall-v2`| 1.5|
  |`sweep-into-v2`| 1.0|
  |`drawer-open-v2`| 1.0|
  |`lever-pull-v2`| 3.0|
  |`handle-pull-side-v2`| 1.0|
  |`peg-insert-side-v2`| 5.0|
  |`peg-unplug-side-v2`| 2.5|
  |`hammer-v2`| 5.0|

  | DMControl Dataset     | data_qualtiy  |
  |-------------|-----|
  |`hopper-hop`| 8.0|
  |`walker-walk`| 1.0|
  |`humanoid-walk`| 7.0|

</details>

<br>



# Setup
## MR (Indepedent pairwise sampling)
```
q_budget=1
model_type=BT
feedback_type=RLT
```

## SeqRank
```
q_budget=1
model_type=BT
feedback_type=SeqRank
```

## LiRE
```
q_budget=100
model_type=linear_BT
feedback_type=RLT
```

# How to run

- Metaworld: [MetaWorld script](./MetaWorld.sh)

- DMControl: [DMControl script](./DMControl.sh)

- Examples: [Example script](./example.sh)

- GT / wrong rewards: [GT script](./GT.sh)
