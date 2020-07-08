## Semantic Correspondence as an Optimal Transport Problem
This is the implementation of our CVPR2020 paper "Semantic Correspondence as an Optimal Transport Problem" by Liu, Y., Zhu, L., Yamada, M. and Yang, Y.

Implemented on Python 3.6 and Pytorch 1.4.0.

For more information, check out the paper on [[CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Semantic_Correspondence_as_an_Optimal_Transport_Problem_CVPR_2020_paper.pdf)].

### Conda environment settings

    conda create -n scot python=3.6
    conda activate scot

    cat /usr/local/cuda/version.txt
    conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch (if CUDA 10) 
    conda install pytorch=1.4.0 torchvision cudatoolkit=9.0 -c pytorch (if CUDA 9) 
    
    conda install -c anaconda scikit-image
    conda install -c anaconda pandas
    conda install -c anaconda requests
    pip install gluoncv-torch

### Reproduction    

   
SCOT Results on PF-PASCAL with res101 (PCK: 63.2/85.4/92.8)

    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.05
    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.10
    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.15

SCOT Results on PF-PASCAL with resnet101-FCN (PCK: 67.1/89.0/95.4)

    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.05
    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.10
    python evaluate_map_CAM.py --dataset pfpascal --thres img --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.15


SCOT Results on PF-WILLOW with res101 (PCK: 48.0/76.2/87.1)

    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.05
    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.10
    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone resnet101 --hyperpixel '(2,22,24,25,27,28,29)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.15


SCOT Results on PF-WILLOW with res101-FCN (PCK: 50.6/78.1/89.1)
    
    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.05
    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.10
    python evaluate_map_CAM.py --dataset pfwillow --thres bbox --backbone fcn101 --hyperpixel '(2,4,5,18,19,20,24,32)' --sim OTGeo --exp1 1.0 --exp2 0.5 --eps 0.05 --gpu 0 --classmap 1 --split test --cam FCN --alpha 0.15


SCOT Results on SPair-71k with res101 no CAM (PCK: 31.3/34.8) 

    python evaluate_map_CAM.py --dataset spair --thres bbox --backbone resnet50 --hyperpixel '(0,11,12,13)' --sim OTGeo --exp1 1.0 --exp2 1.0 --eps 0.05 --gpu 0 --classmap 0 --split test --alpha 0.10
    python evaluate_map_CAM.py --dataset spair --thres bbox --backbone resnet101 --hyperpixel '(0,19,27,28,29,30)' --sim OTGeo --exp1 1.0 --exp2 1.0 --eps 0.05 --gpu 0 --classmap 0 --split test --alpha 0.10


SCOT Results on SPair-71k with res101 (PCK: 32.1/35.4)
    
    python evaluate_map_CAM.py --dataset spair --thres bbox --backbone resnet50 --hyperpixel '(0,11,12,13)' --sim OTGeo --exp1 1.0 --exp2 1.0 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.10
    python evaluate_map_CAM.py --dataset spair --thres bbox --backbone resnet101 --hyperpixel '(0,19,27,28,29,30)' --sim OTGeo --exp1 1.0 --exp2 1.0 --eps 0.05 --gpu 0 --classmap 1 --split test --alpha 0.10


SCOT on TSS with res101

    python evaluate_map_TSS_CAM.py --dataset TSS --thres img --backbone resnet101 --hyperpixel '(2, 22, 24, 25, 27, 28, 29)' --sim OTGeo --exp1 1 --exp2 1 --eps 0.05 --gpu 0 --classmap 1


Beam search 
    
    python beamsearch.py --dataset pfpascal --backbone resnet101 --thres img --exp1 1.0 --exp2 0.5 --classmap 0
    python beamsearch.py --dataset spair --backbone resnet50 --thres bbox --classmap 0
    python beamsearch.py --dataset spair --backbone resnet101 --thres bbox --classmap 0

    
### Bibtex
If you use this code or results for your research, please consider citing:
````
@inproceedings{liu2020semantic,
    title={Semantic Correspondence as an Optimal Transport Problem},
    author={Liu, Yanbin and Zhu, Linchao and Yamada, Makoto and Yang, Yi},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={4463--4472},
    year={2020}
}
````
