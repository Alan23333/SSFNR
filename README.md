# Getting started

## Setup environment

1. clone the repository

```bash
git clone git@github.com:WangXueyang-uestc/ARConv.git
cd ARConv
install dependencies

bash
复制代码
pip install -r requirements.txt
Prepare dataset
Datasets can be downloaded from the repo liangjiandeng/PanCollection. Remember to replace PATH TO TRAIN DATASET in .scripts with the path to the dataset.

Train the model
bash
复制代码
bash .scripts/train_{Datasets}.sh
Additional Information
Weights:

You can downloaded our trained weights from https://pan.baidu.com/s/1xFWSLX9611E2CukCpnOp1w?pwd=37ic.

Metrics:

MetricCode can be found here.

You can also use the tools from liangjiandeng/DLPan-Toolbox (specifically, the 02-Test-toolbox-for-traditional-and-DL(Matlab) directory).
