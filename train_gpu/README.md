# The training part

---
## Environment
---
- Ubuntu18.04 and Tensorflow-gpu version (1.15.0)

```bash
conda create -n mknet python=3.7
conda activate mknet
pip install tensorflow-gpu==1.15.0 protobuf==3.20.0 h5py plyfile 
git clone https://github.com/himlen1990/MKAnet.git
```

## Generate a simple training dataset
```bash
cd MKANet/utils
python scene_generation.py
cp scene_bottlecup.h5 ../train_gpu/demo
```

## Compile tensorflow operator
cd <your anaconda root>/envs/mknet/lib/python3.7/site-packages/tensorflow_core
ln -s libtensorflow_framework.so.1 libtensorflow_framework.so

```bash
cd ../train_gpu/utils/tf_ops/
sh compile.sh
```

## Training the network
```bash
python train.py
```