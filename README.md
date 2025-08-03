## 1. Create Environment

We suggest you use pytorch 1.11 to re-implement the results in our ICCV 2023 paper and pytorch 2 to re-implement the results in NTIRE 2024 Challenge because pytorch 2 can save more memory in mix-precision training.

### 1.1 Install the environment with Pytorch 1.11

- Make Conda Environment
```
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

### 1.2 Install the environment with Pytorch 2

- Make Conda Environment
```
conda create -n torch2 python=3.9 -y
conda activate torch2
```

- Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

&nbsp;

## 2. Prepare Dataset
Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w) (code: `gplv`), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)

SMID [Baidu Disk](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ) (code: `btux`), [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)

SDSD-indoor [Baidu Disk](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F) (code: `jo1v`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

SDSD-outdoor [Baidu Disk](https://pan.baidu.com/share/init?surl=JzDQnFov-u6aBPPgjSzSxQ) (code: `uibk`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

MIT-Adobe FiveK [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing), [Official](https://data.csail.mit.edu/graphics/fivek/)

NTIRE 2024 [Baidu Disk](https://pan.baidu.com/s/1Tl-LUhwsPh6XFA2SqR5c8Q?pwd=cyh2) (code:`cyh2`), Google Drive links for [training input](https://drive.google.com/file/d/1Js9yHmV0xAWhT5oJKzfx6oOr_7k5hcNg/view), [training GT](https://drive.google.com/file/d/1PUJgJiEyrIj5TgwcQlFvVGuIe3_PXMLY/view), and [mini-val set](https://drive.google.com/drive/folders/1M-WVWToH1HhMtmQlYrb8qlNCQgi0kG3y?usp=sharing).

**Note:** 

(1) Please use [bandizip](https://www.bandisoft.com/bandizip/) to jointly unzip the `.zip` and `.z01` files of SMID, SDSD-indoor, and SDSD-outdoor 

(2) Please process the raw images of the MIT Adobe FiveK dataset following [the sRGB output mode](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38) or directly download and use the sRGB image pairs processed by us in the [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`) and [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing)

(3) Please download the `text_list.txt` from [Google Drive](https://drive.google.com/file/d/199qrfizUeZfgq3qVjrM74mZ_nlacgwiP/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1GQfaQLI6tvB0IrTMPOM_9Q?pwd=ggbh) (code: `ggbh`) and then put it into the folder `data/SMID/SMID_Long_np/`
