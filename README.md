# moral-driving-agent
Moral Driving Agent

Note: All scripts should be executed from `src/` directory

## Moral Rewards
Extract data from Moral Machine Dataset
```
python -m moral_rewards.scripts.extract_moral_data \
--input_data_path 'data/SharedResponses.csv' \
--output_data_path 'data/moral_data.csv' \
--countries 'SGP' \
--chunksize 100000
```

Convert data from extracted Moral Machine Dataset
```
python -m moral_rewards.scripts.convert_moral_data \
--input_data_path 'data/moral_data.csv' \
--data_path_train 'data/moral_data_train.npz' \
--data_path_val 'data/moral_data_val.npz' \
--data_path_test 'data/moral_data_test.npz' \
--train_size 0.8 \
--val_size 0.1 \
--random_seed 0
```
