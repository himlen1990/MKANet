# Utils for dataset generation and result visualization

## Environment
-pcl1.7 or 1.8(if your pcl version is 1.8, modify the CMakeLists)

## Install
```bash
cd build
cmake ..
make
```

## Dataset generation
```bash
python scene_generation.py
```
- In this demo, we only have two function category, you can modify the source code to fit your dataset.

## Check the generated results
recover_h5_to_ply.py scene_name frame_num
example: recover_h5_to_ply.py scene_bottlecup.h5 0

## Trajectory Visualization
place the ply and txt files to the build folder
```bash
./show_traj predicted.ply predicted.txt
```

##Annotation
- If you want to create your annotation, you can use the following annotation tool
https://github.com/himlen1990/toolbox/tree/master/annotation_tool

## Notice
When creating your custom dataset, make sure the function points are set to 255 in the blue channel and the no function points are set to 255 is the red channel (a pouring example is given in dataset/train_data)