#/bin/bash
CUDA_ROOT=/usr/local/cuda
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $CUDA_ROOT
echo $TF_INC
echo $TF_LIB

# TF>=1.4.0
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
