# JCTNet

Joint CNN and Transformer Network via weakly supervised Learning for efficient crowd counting.

[Arxiv]([https://arxiv.org/pdf/2203.06388])

We propose a Joint CNN and Transformer Network (JCTNet) via weakly supervised learning for crowd counting in this paper. JCTNet consists of three parts: CNN feature extraction module (CFM), Transformer feature extraction module (TFM), and counting regression module (CRM). In particular, the CFM extracts crowd semantic information features, then sends their patch partitions to TRM for modeling global context, and CRM is used to predict the number of people. Extensive experiments and visualizations demonstrate that JCTNet can effectively focus on the crowd regions and obtain superior weakly supervised counting performance on five mainstream datasets. The number of parameters of the model can be reduced by about 67%∼73% compared with the pure Transformer works. We also tried to explain the phenomenon that a model constrained only by countlevel annotations can still focus on the crowd regions. We believe our work can promote further research in this field.


## Prerequisites

Python 3.x

Pytorch >= 1.2

For other libraries, check requirements.txt.

## NetWork FrameWork
!()[]


## Getting Started
1. Dataset download

  + QNRF can be downloaded [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)

  + NWPU can be downloaded [here](https://www.crowdbenchmark.com/nwpucrowd.html)

  + Shanghai Tech Part A and Part B can be downloaded [here](https://www.kaggle.com/tthien/shanghaitech)

2. Data preprocess

  Due to large sizes of images in QNRF and NWPU datasets, we preprocess these two datasets.
  ```
  python preprocess_dataset.py --dataset <dataset name: qnrf or nwpu> --input-dataset-path <original data directory> --output-dataset-path <processed data directory>

  ```
3. Training
  ```
  python train_JCTNet.py --data-dir <path to dataset> --dataset <dataset name: qnrf, sha, shb or nwpu> --max-epoch 2000 --batch-size 32 --device 0 --crop-size 256

  ```
4. Test
  ```
  python test_image_patch.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset> --dataset <dataset name: qnrf, sha, shb or nwpu>
  ```

5. vis
  visualize predicted densitymap.
  ```
  python vis_densityMap.py --device <device_id> --image_path part_A_final/test_data/images/IMG_104.jpg --weight_path JCTnet/ckpts/SHA_best_model_mae 62.20.pth
  ```
  visualize heatmap，modified code-column116/117.
  ```
  python vis_grad_cam.py 
  ```
6. Pretrained models
  Pretrained models on UCF-QNRF, NWPU, Shanghaitech part A and B can be found [Google Drive](https://drive.google.com/drive/folders/10U7F4iW_aPICM5-qJq21SXLLkzlum9tX?usp=sharing). You could download them and put them in in pretrained_models folder.

7. JCTNet weights pth in ShanghaiTechA
  weight_path = ./ckpts/SHA_best_model_mae 62.20.pth
  ```
  python test_image_patch.py --model-path ./ckpts/SHA_best_model_mae 62.20.pth --data-path <directory for the ShanghaiTech_A dataset> --dataset sha
  ```



## References

If you find this work or code useful, please cite:

```
@inproceedings{Wang F, Liu K, Long F, et al. Joint cnn and transformer network via weakly supervised learning for efficient crowd counting[J]. arXiv preprint arXiv:2203.06388, 2022.
}
```

```

```
