<h1 align="center"> Diffmv: A Unified Diffusion Framework for Healthcare Predictions with Random Missing Views and View Laziness </h1>


## About Our Work

Update: 2025/05/17: We have created a repository for the paper titled *Diffmv: A Unified Diffusion Framework for Healthcare Predictions with Random Missing Views and View Laziness*, which has been accepted to the *SIGKDD 2025*. In this repository, we offer the preprocessing scripts and algorithm files to showcase the reproducibility of our work.

![image-20250111103132521](https://s2.loli.net/2025/01/11/5ZKURGucnmApWeJ.png)

![image-20250111103154909](https://s2.loli.net/2025/01/11/AGVkwu2S3LXg5j4.png)

## Requirements

- openai==1.3.5
- torch==1.13.1+cu117
- dgl==1.1.2
- pyhealth==1.1.4
- seaborn==0.13.0

## Data Sets

Owing to the copyright stipulations associated with the dataset, we are unable to provide direct upload access. However, it can be readily obtained by downloading directly from the official website: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/),[eICU](https://eicu-crd.mit.edu/). 

The structure of the data set should be like,

```powershell
data
|_ PHE
|  |_ MIII
|  |_ _processed
|  |_ _ _datasets_pre_stand.pkl
|  |_ MIV-Note
|  |_ _ _datasets_pre_stand.pkl
|_ LOS
|_ _MIII
|  |_ _processed
|  |_ _ _datasets_pre_stand.pkl
|  |_ _ rare_patient.pkl
```

## RUN

```powershell
# run the main file
# change config.py
config = {**vars(PHECONFIG), **PHECONFIG.get_params()}
config = {k: v for k, v in config.items() if not k.startswith('__')}


# please set pretrain=True (first, sencod stage) 
# please set tuning=True (third stage)
python main_unify.py
```

## Acknowledge & Contact

You could contact czhaobo@connect.ust.hk if you have any problems. If you want to make several collaboration in healthcare fields, please do not hesitate to send an email to him.

