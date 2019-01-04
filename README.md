## In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
Yuezun Li, Ming-ching Chang and Siwei Lyu \
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

### Usage
#### Toy with VGG16 network
1. Download pretrained models: [CNN-VGG16](https://drive.google.com/drive/folders/1gACZmcVuHL48DDCWUawxDTkSqLcLjM_j?usp=sharing)
and put the model into `ckpt_CNN`.
2. Go to `toy` folder and run `run_cnn.py` with arguments as following. 
 ```
  python run_cnn.py \
  --input_vid_path=/path/to/toy_video \
  --cache_dir=where_to_save_cache \
  --out_dir=where_to_save_output
  ```

#### Toy with LRCN-VGG16 network
1. Download pretrained model [LRCN-VGG16](https://drive.google.com/drive/folders/1gACZmcVuHL48DDCWUawxDTkSqLcLjM_j?usp=sharing) and put the model into `ckpt_LRCN`.
2. Go to `toy` folder and run `run_cnn.py` with arguments as following. 
 ```Shell
  python run_lrcn.py \
  --input_vid_path=/path/to/toy_video \
  --cache_dir=where_to_save_cache \
  --out_dir=where_to_save_output
  ```
The probability of eye state will be put in `.p` file and a plot video will be generated.

### Train and Test
Under construction...
### Citation

Please cite our paper in your publications if it helps your research:

    @inproceedings{li2018ictu,
      title={In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking},
      author={Li, Yuezun and Chang, Ming-Ching and Lyu, Siwei},
      Booktitle={IEEE International Workshop on Information Forensics and Security (WIFS)},
      year={2018}
    }
