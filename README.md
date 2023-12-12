# H-GAP: Humanoid Control with a Generalist Planner

Implementation of [H-GAP: Humanoid Control with a Generalist Planner](https://yingchenxu.com/hgap/).


## Installation
1. Create a Python virtual environment with Python 3.9 via the method of your choice. For example with conda:
```
conda create -n hgap python=3.9
```

2. Install the dependencies:
```
pip install -r requirements/requirements.txt
```

3. Install MoCapAct following instruction at https://github.com/microsoft/MoCapAct

## Prepare MoCapAct datasets

1. Download MoCapAct datasets following instructions at https://github.com/microsoft/MoCapAct

2. Generate TFDS datasets for H-GAP training:

```
# This creates TFDS datasets from the original MoCapAct dataset (which is in HDF5 format).
# Set mocapact_data_dir to be the path to the downloaded MoCapAct dataset, e.g. /home/usr/data/mocap
# Note that there are two sizes of MoCapAct, i.e. small and large. Specify which one to build by setting size as small or large.

python trajectory/datasets/generate_tfds_dataset.py --mocapact_data_dir $mocapact_data_dir --size small
```

## Usage

1. Train VAE:
```
python scripts/train.py --dataset mocapact-large-compact --exp_name $vae_name --relabel_type none --n_epochs_ref 1200
```

2. Train Prior Transformer:
```
python scripts/trainprior.py --dataset mocapact-large-compact --exp_name $prior_name --vae_name $vae_name --relabel_type none --n_epochs_ref 1200 

```

3. Plan:
```
python scripts/humanoid_plan.py --test_planner sample_with_prior --objective $relabel_type --temperature 2 --prob_weight 0 --nb_samples 64 --horizon 16 --dataset mocapact-large-compact --exp_name $plan_name --prior_name $prior_name --vae_name $vae_name --suffix $j --seed $j --task $relabel_type  --top_p $top_p
```

## License
The majority of H-GAP is licensed under CC-BY-NC, however portions of the project are adapted from codes available under separate license terms: latentplan is licensed under the MIT license.
