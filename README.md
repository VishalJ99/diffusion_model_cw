**********************************************
# M2 CW: Training Diffusion Models With Custom Degradation
**********************************************

## Description
Project for M2 course work, training diffusion models
with custom degradations.

## Installation
Set up
```
git clone git@github.com:VishalJ99/diffusion_model_cw.git

cd diffusion_model_cw

conda env create -f environment.yml

conda activate m2_cw_env

docker build -t m2_docker .
```
## Usage
All jobs will create a folder in the `runs` directory (which will also be created if it doesn't exist) with the job name set in the config. The folder will contain the following:
```
job_name/
├── model_weights/
│   ├── best_model.pth
│   └── ... (other model weight files)
├── samples/
│   ├── conditional/
│   │   └── ... (conditional samples)
│   └── unconditional/
│       └── ... (unconditional samples)
├── losses.csv
└── train_config.yaml
```

To run any of the following commands via docker use:

Docker:

```
docker run -it -v $(pwd):/m2_cw m2_docker /bin/bash -c "source activate m2_cw_env && {command}"
```

Eg. to run q1 good train job, which would be locally run as:

```
python src/train.py configs/q1_good_train_config.yaml
```
This can be run in docker as:

```
docker run -it -v $(pwd):/m2_cw m2_docker /bin/bash -c "source activate m2_cw_env && python src/train.py configs/q1_good_train_config.yaml"
```

### Training

NOTE: These jobs will take a long time to run, especially on a CPU. It is recommended to run these jobs on a device with a CUDA GPU or with MPS atleast. If running in such an environment, docker commands are not advised as they will not be able to access the GPU currently.

To speed up training, you can reduce the number of epochs in the config files or run with quick_train set to True. If docker commands are run, just be aware on the cpu generating the conditional and unconditional sample plots at the end of every epoch can take 5-10 minutes.

Ensure environment is activated if running on a local machine.



#### Commands
q1 good train job:
```
python src/train.py configs/q1_good_train_config.yaml
```

q1 bad train job:
```
python src/train.py configs/q1_bad_train_config.yaml
```


q2 ddpm unet train job:
```
python src/train.py configs/q2_ddpm_train_config.yaml
```

q2 gaussian blur train job:

```
python src/train.py configs/q2_gauss_blur_train_config.yaml
```
NOTE: The conditional sample plots will look different from the ones presented in the report as by default the plots are made with normalise=True, this option is not configurable from the config currently and should be changed in train.py if needed.

Change line 116 in `src/train.py` from:

```
plt = make_cond_samples_plot(z_t, visualise_ts, num_cond_samples)
```

to:

```
plt = make_cond_samples_plot(z_t, visualise_ts, num_cond_samples, normalise=False, standardise=True)
```



### Testing
Takes about 5-10 minutes a job on an M1 Macbook Pro, since it needs to generate 1000 samples conditionally to compute the image quality metrics. Can turn this off by setting `calc_metrics` to False or decreasing the `metric_sample_size` in the config file.

These will write loss and metrics to the csv file in the job folder.

q1 good test job:
```
python src/test.py configs/q1_good_test_config.yaml
```

q1 bad test job:
```
python src/test.py configs/q1_bad_test_config.yaml
```

q2 ddpm unet test job:
```
python src/test.py configs/q2_ddpm_test_config.yaml
```

q2 gaussian blur test job:
```
python src/test.py configs/q2_gauss_blur_test_config.yaml
```


## License
Please see license.md

## Author
Vishal Jain
2024-03-12
