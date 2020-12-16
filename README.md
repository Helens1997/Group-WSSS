## Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation

This  is a PyTorch implementation of our group-wise learning framework for weakly supervised semantic segmentation, which is accepted in AAAI 2021.

**Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation.** [[arXiv](http://arxiv.org/abs/2012.05007)]

*Xueyi Li, Tianfei Zhou, Jianwu Li, Yi Zhou and Zhaoxiang Zhang.* AAAI 2021.


## Prerequisites

We train the model using PyTorch 1.4.0 with four NVIDIA RTX 2080Ti GPU with 11GB memory per card.

- [PyTorch 1.4.0]((https://github.com/pytorch/pytorch))

Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
``` 

## Training

### Clone

```git clone -- recursive https://github.com/Lixy1997/Group-WSSS```

### Prepare Dataset

In the paper, we use PASCAL VOC 2012 for training. Here are the steps to prepare the data:

1. Download the [PASCAL VOC 2012](https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view) dataset.

2. Create soft links:

    ```cd data; ln -s your/path VOC2012;```

### Stage #1: Train the classification network for group-wise semantic mining

1. Once the data is prepared, please run ```python train.py``` for training the classification network with our default parameters.

    After the network is finished, you can resize the maps to the original image size by

    ```bash
    cd run/pascal
    python res.py
    ``` 
2. Move the resized maps to the ```data/VOCdevkit/VOC2012/``` folder.

   Put the saliency maps to the ```data/VOCdevkit/VOC2012/``` folder, or you can run DSS model to generate saliency maps by yourself.

3. Generate the pseudo labels of the training set by

    ```bash
    python gen_labels.py
    ```

### Stage #2: Train the semantic segmentation network

Once the pseudo ground-truths are generated, they are employed to train the semantic segmentation network. We use Deeplab-v2 in all experiments. But most popular FCN-like segmentation networks can be used instead.  

## Our Results

1. The visualization of CAMs generated by our group-wise semantic mining can be downloaded from [Google Drive](https://drive.google.com/file/d/1o7zqOwGKmUtR2VS5i30xLovIZI9vFm3b/view?usp=sharing).

2. The saliency maps used as pseudo labels can be downloaded from [Google Drive](https://drive.google.com/file/d/1Ls2HBtg3jUiuk3WUuMtdUOVUFCgvE8IX/view).

3. The pseudo ground-truths of PASCAL VOC 2012 generated by our model can be download from [Google Drive](https://drive.google.com/file/d/1ICjerndySg5-KWbXFol9O8jmbyIz7by3/view?usp=sharing)

4. The segmentation results of val and test sets of PASCAL VOC 2012 dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1bm8zmrMPSXbptg9ANuWZm5ed6oQDSHvy/view?usp=sharing).
For reproducing scores of the test set, please submit the results of test set to the [official website](http://host.robots.ox.ac.uk:8080/) following the instructions of the official website.


## Citation
If you find this work useful for your research, please consider citing the following paper:
```
@article{li2020group,
  title={Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation},
  author={Li, Xueyi and Zhou, Tianfei and Li, Jianwu and Zhou, Yi and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2012.05007},
  year={2020}
}
```

