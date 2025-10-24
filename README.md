# Getting started

## ⚙️ Setup environment

1. clone the repository

```bash
git clone https://github.com/Alan23333/SSFNR.git
cd SSFNR
```

2. Install Python environment

```bash
conda create -n your_env_name python=3.8.20
```

3. install dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Prepare dataset

Datasets can be downloaded from: [data](https://pan.baidu.com/s/12GSZavmQdkVf3rI6OoylNA?pwd=1pbh)


## 🧱 Dataset Directory Structure

Your dataset should be organized as follows under the `data/` directory:

```

data/
├── CAVE/
│   ├── complete_ms_data/
│   ├── f4/
│   ├── f8/
│   └── response coefficient.mat
├── Harvard/
├── Chikusei/
├── Pavia/

```

## 🛠️ Data Preprocessing

Before training, you need to preprocess the data.

1. **Set the `--prepare` flag to `Yes`** in `option.py` to trigger dataset preprocessing.

2. **Run the main script**:
   ```bash
   python main.py


## 🚀 Train the model

```bash
python main.py
```


### 📦 Weights

You can download our trained weights from:
[https://pan.baidu.com/s/1xFWSLX9611E2CukCpnOp1w?pwd=37ic](https://pan.baidu.com/s/1xFWSLX9611E2CukCpnOp1w?pwd=37ic)
[https://pan.baidu.com/s/12GSZavmQdkVf3rI6OoylNA?pwd=1pbh]

Password: `37ic`


  [https://github.com/liangjiandeng/DLPan-Toolbox/tree/master/02-Test-toolbox-for-traditional-and-DL(Matlab)](https://github.com/liangjiandeng/DLPan-Toolbox/tree/master/02-Test-toolbox-for-traditional-and-DL%28Matlab%29)
* You can also use the tools from **liangjiandeng/DLPan-Toolbox** (specifically, the `02-Test-toolbox-for-traditional-and-DL(Matlab)` directory):
  [https://github.com/liangjiandeng/DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox)



## 🧪 Testing

To perform testing using a pre-trained model, follow the steps below:

1. **Edit `option.py`**

- Set the `test_only` argument to `store_false` to enable **testing mode**:
- Set the `pre_train` argument to point to the model you want to test.

2. **Run the main script**:

```bash
python main.py
```



