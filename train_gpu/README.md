# The training part

---
## Environment
---
- Ubuntu14.04/16.04 and Tensorflow-gpu version (1.4 - 1.11) (docker is recommended)

## Docker Setup
- Create docker image
```bash
docker pull nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
```
-Create docker container
```bash
docker run -it --gpus all nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
```
-Inside docker container, run
```bash
apt-get update
apt-get install python python-pip git 
pip install numpy==1.16.2 mock==2.0.0 h5py==2.9.0 Markdown==3.0.1 tensorflow-gpu==1.4.0 plyfile h5py
cd
git clone https://github.com/himlen1990/MKAnet.git
```

## Generate a simple training dataset
```bash
cd MKANet/utils
python scene_generation.py
cp scene_bottlecup.h5 ../train_gpu/demo
```

## Compile tensorflow operator
```bash
cd ../train_gpu/utils/tf_ops/
sh compile.sh
```

## Training the network
```bash
python train.py
```