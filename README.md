## In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
Yuezun Li, Ming-ching Chang and Siwei Lyu \
University at Albany, State University of New York, USA \
IEEE International Workshop on Information Forensics and Security (WIFS), 2018 \
[https://arxiv.org/abs/1806.02877](https://arxiv.org/abs/1806.02877)


### Contents
1. [Requirement](#Requirement)
2. [Usage](#Usage)
3. [Train](#Train)


### Requirement
- Python 2.7
- Ubuntu 16.04
- Tensorflow 1.3.0
- CUDA 8.0
```bash
Required Python packages 
yaml==3.12
easydict==1.7
matplotlib==1.5.3
dlib==19.16.0
opencv==3.4.0
tqdm==4.19.5
```
### Docker Setup (Just run the docker with all the packages already installed)
## Install Docker
```
sudo apt install docker.io
```
## Install Nvidia Toolkit
```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
## Pull docker image and run
```
sudo docker pull harshhhhteen/mytense
sudo docker run -it --gpus all -v `pwd`/WIFS2018_In_Ictu_Oculi-master:/WIFS2018_In_Ictu_Oculi-master harshhhhteen/mytense bash
# In the container location of `toy` directory `/WIFS2018_In_Ictu_Oculi-master/toy`
```

### Usage
#### Toy with VGG16 network
1. Download pretrained models: [CNN-VGG16](https://1drv.ms/u/s!As4tun0sWfKsgdJKjz0aPoaPy1QSoA?e=Ba0gHC)
and put the model into `ckpt_CNN`.
2. Go to `toy` folder and run `run_cnn.py` with arguments as following. 
 ```
  python run_cnn.py \
  --input_vid_path=/path/to/toy_video \
  --out_dir=where_to_save_output
  ```

#### Toy with LRCN-VGG16 network
1. Download pretrained model [LRCN-VGG16](https://1drv.ms/u/s!As4tun0sWfKsgdJKjz0aPoaPy1QSoA?e=Ba0gHC) and put the model into `ckpt_LRCN`.
2. Go to `toy` folder and run `run_cnn.py` with arguments as following. 
 ```Shell
  python run_lrcn.py \
  --input_vid_path=/path/to/toy_video \
  --out_dir=where_to_save_output
  ```
The probability of eye state will be put in `.p` file and a plot video will be generated.

#### UADFV dataset 
Fill out the [Google form](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform) or [Tencent form](https://wj.qq.com/s2/13352936/6b20/) to get the download link.

#### EBV dataset
Download [here](https://1drv.ms/u/s!As4tun0sWfKsgdVcYJ-nn0bw0kdjzw?e=DAcGfb).

### Train
1. `train_blink_cnn.py` and `train_blink_lrcn.py` are training scripts for CNN and LRCN respectively.
2. `proc_data` contains the data preparation process for training CNN and LRCN.
3. `sample_eye_data` contains images for training CNN, `sample_sq_data` contains sequences for training LRCN.
We collect many videos from Internet and manually annotate the eye state of each frame. 
Due to the copyright issue, the collected set is not published. Thus I only upload an example in each folder. 
### Citation

Please cite our paper in your publications if it helps your research:

    @inproceedings{li2018ictu,
      title={In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking},
      author={Li, Yuezun and Chang, Ming-Ching and Lyu, Siwei},
      Booktitle={IEEE International Workshop on Information Forensics and Security (WIFS)},
      year={2018}
    }

#### Notice
This repository is NOT for commecial use. It is provided "as it is" and we are not responsible for any subsequence of using this code.
