**********************************************
# M2 CW: Training Diffusion Models With Custom Degradation
**********************************************

## Description
Project for M2 course work, training diffusion models 
with custom degradations.

## Installation
Set up
```
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/M2_Assessment/vj279.git

cd vj279

conda env create -f environment.yml

conda activate m2_cw_env
```
## Usage
All jobs will create a folder in the `runs` directory with the job name and a timestamp. The folder will contain the following:
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

### Local Usage
(Preferred if using macOS with apple silicon) 

Ensure environment is activated

### Training

NOTE: These jobs will take a long time to run, especially on a CPU. It is recommended to run these jobs on a GPU or with MPS atleast. To speed up training, you can reduce the number of epochs in the config files or run with quick_train set to True.
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
python src/train.py configs/q2_ddpm_unet_train_config.yaml 
```

q2 gaussian blur train job:
```
python src/train.py configs/q2_gaussian_blur_train_config.yaml 
```

### Testing

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
python src/test.py configs/q2_ddpm_unet_test_config.yaml 
```

q2 gaussian blur test job:
```
python src/test.py configs/q2_gaussian_blur_test_config.yaml 
```


## License
Please see license.md

## Author
Vishal Jain
2024-03-12
