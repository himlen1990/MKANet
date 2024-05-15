# The depoly part

- Notice that you do not need a gpu in the depoly phase, so you can depoly the trained result in your notebook.

---
## Environment
---
- Tensorflow 1.4 (running within anaconda is recommended)

```bash
pip install tensorflow==1.15.0 protobuf==3.20.0 matplotlib plyfile h5py
```

## Compile tensorflow operator in CPU ver
```bash
cd <your anaconda root>/envs/teapot/lib/python3.7/site-packages/tensorflow_core
ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
cd <...>/MKANet/deploy_cpu/
sh tf_complie.sh
```

## Deploy
- Replace the log folder with you trained result.
```bash
python test.py test_data/1.ply
```

## Result visualization
- Check the MKANet/utils directory
