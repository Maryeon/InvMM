# InvMM

Official PyTorch implementation of our paper [An Inversion-based Measure of Memorization for Diffusion Models](An Inversion-based Measure of Memorization for Diffusion Models). Note that we are actively improving our paper (till it get published), so that there might be some inconsistence between this updated repository and the results presented in the old version of paper.

![teaser](assets/teaser.png)

This repository follows the implemetations of codebases [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [stable-diffusion-v1](https://github.com/CompVis/stable-diffusion), [stable-diffusion-v2](https://github.com/Stability-AI/stablediffusion) and [stable-diffusion-v3](https://github.com/Stability-AI/sd3.5). *We also modify some source codes, including introducing [xformers](https://github.com/facebookresearch/xformers) support for stable-diffusion-v1 and fixing bugs in DDIM sampler.*

## Requirements

Go to the specific directory and create an anaconda environment with:

```shell
cd ddpm[latent-diffusion, stable-diffusion-v1, stable-diffusion-v2, stable-diffusion-v3]
conda env create -f environment.yaml
```

Change Torch and xformers to appropriate versions depending on your own CUDA run time library.

## Data Preparation

### CIFAR-10

CIFAR-10 can be downloaded on the [official website](https://www.cs.toronto.edu/~kriz/cifar.html). Obtain the IDs of [99 highly memorized images](https://drive.google.com/file/d/1pFbNl8kiK77NFeaNXo4Qdx3eSaNulEcx/view?usp=sharing) and [1000 normal images](https://drive.google.com/file/d/18YA-PW8jtrpupUYUXX8rac0qw_Ib5mdN/view?usp=sharing) if needed.

### Faces

We use the [CelebAHQ-256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) dataset on Kaggle and FFHQ following their [official repository](https://github.com/NVlabs/ffhq-dataset).

### LAION

The subsets of LAION used in the paper can be downloaded [here](https://drive.google.com/file/d/1M_rCjEz8w0JeYI7v2Aekp514Iav5SyJp/view?usp=sharing).

## Pretrained Distribution Parameters

| Model   | Dataset       | Link                                                         |
| :------ | :------------ | :----------------------------------------------------------- |
| DDPM    | CIFAR-10      | https://drive.google.com/file/d/1TJDmFdb6-ZwI2AqOfTCClNWn_iAGjdvN/view?usp=sharing |
| LDM     | CelebAHQ FFHQ | https://drive.google.com/drive/folders/1eeO9E4zLTdy1PfPA55YhIwclS9XBF-UI?usp=sharing |
| SD v1.4 | LAION Subset  | https://drive.google.com/drive/folders/1TNvSc6JMvCqZJ4-9FO-A4-bwjYVReOIc?usp=sharing |
| SD v2.1 | LAION Subset  | https://drive.google.com/drive/folders/1qiFMpUfLdZdLWRV-TkmPJ1-AEMmUsF07?usp=sharing |
| SD v3.5 | LAION Subset  | https://drive.google.com/drive/folders/1bjXDH8iQOb5F-ADBBModB_vmIMrGbhvg?usp=sharing |

## Inversion

We use SSCD to calculate image similarity. Download the ```sscd_disc_large``` model in their [official repository](https://github.com/facebookresearch/sscd-copy-detection).

### DDPM

Download [our pretrained DDPM](https://drive.google.com/file/d/1ktZzkNMGiKlNjMA05dcD_0ehp3laRCp9/view?usp=sharing) and run the following command to perform inversion and calculate memorization scores:

```shell
python inversion.py --logdir logs/DDPM_CIFAR10_EPS_INVERSION
```

### Latent Diffusion

We provide [pretrained models](https://drive.google.com/drive/folders/1jVt9oUOJ2Z32XA_oirAjl3Pb40SsTkNT?usp=sharing) on the subsets of CelebAHQ and FFHQ. Pretrained models on the full datasets can be found in the official repository.

To perform inversion:

```shell
python inversion -dp /path/to/dataset --ckpt_file /path/to/pretrained_model
```

### Stable Diffusion

Download pretrained [SD v1.4](https://github.com/CompVis/stable-diffusion), [SD v2.1](https://github.com/Stability-AI/stablediffusion) and [SD v3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/tree/main), and then run:

```shell
python inversion -dp /path/to/dataset
```

## LICENSE

Each model follows their original license.
