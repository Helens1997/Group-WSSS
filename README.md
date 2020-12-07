## Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation

This  is a PyTorch implementation of our model for weakly supervised semantic segmentation.

**Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation.**[[Arxiv]()]

## Prerequisites

The training experiments are conducted using PyTorch 1.4.0 with four NVIDIA RTX 2080Ti GPU with 11GB memory per card.

+ Install PyTorch 1.4.0

Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
``` 

## Train

### Clone

```git clone -- recursive https://github.com/Lixy1997/Group-WSSS```

### Download Dataset

In the paper, we use the most popular dataset for training. Here are the steps to prepare the data:

+ Download the [PASCAL VOC 2012](https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view) dataset.

+ Create soft links:

    ```cd data; ln -s your/path VOC2012;```

### Train the Group-Wise Semantic Mining Network

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

### Train the Semantic Segmentation Network

Once the [pseudo labels](https://drive.google.com/file/d/1ICjerndySg5-KWbXFol9O8jmbyIz7by3/view?usp=sharing) are generated, it can be used to train the semantic segmentation network.

## The Results

1. The Class Activation Maps for our group-wise semantic mining can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LkgKGtFP4_lCMnG2BxhzC7kcrBotog3O).

2. The saliency maps used for pseudo labels are from [Google Drive](https://drive.google.com/file/d/1Ls2HBtg3jUiuk3WUuMtdUOVUFCgvE8IX/view).

3. The segmentation results of val and test sets of PASCAL VOC 2012 dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1o7zqOwGKmUtR2VS5i30xLovIZI9vFm3b/view?usp=sharing).
For reproducing scores of the test set, please submit the results of test set to the [official website](http://host.robots.ox.ac.uk:8080/) following the requirements of the official website.


## Citation
If you find group-wsss useful for your research, please consider citing the following paper:
```
@inproceedings{li2021group,
 title = {Group-Wise Semantic Mining for Weakly Supervised Semantic Segmentation},
 author = {Li, Xueyi and Zhou, Tianfei and Li, Jianwu and Zhou, Yi and Zhang, Zhaoxiang},
 booktitle = AAAI,
 year = {2021},
}
```

