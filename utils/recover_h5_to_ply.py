from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import random
import h5py
import glob
import os
import math
from scipy.spatial.transform import Rotation as R
import sys

def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red','uint8'),('green','uint8'),('blue','uint8')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2],pc[i][3], pc[i][4], pc[i][5])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])],text=True)
    ply_out.write(filename)


def recover_data(pc,label, support=False, three_state=True):
    
    fake_color = np.zeros((pc.shape[0],1))
    recover_pc = np.c_[pc,fake_color]
    export_ply(recover_pc,"output_origin.ply")
    obj1 = recover_pc[(pc[:,3]==1).nonzero()]
    obj2 = recover_pc[(pc[:,3]==0).nonzero()]
    if support:
        obj2[:,3] = 255
    else:
        obj1[:,3] = 255
    obj1_function_part = obj1[(obj1[:,4] > 0).nonzero()][:,0:3]
    obj2_function_part = obj2[(obj2[:,4] > 0).nonzero()][:,0:3]
    output_pc = np.r_[obj1,obj2]
    func = output_pc[:,4]
    export_ply(output_pc,"output.ply")

    centroid1 = np.mean(obj1_function_part, axis=0)
    centroid2 = np.mean(obj2_function_part, axis=0)

    if three_state:
        if not support:
            label[:1,:3] =  label[:1,:3]+centroid1
            label[1:,:3] =  label[1:,:3]+centroid2
        else:
            label[:2,:3] =  label[:2,:3]+centroid1
            label[2:,:3] =  label[2:,:3]+centroid2
            label = label[[1,2,0]]
    else:
        label[:2,:3] =  label[:2,:3]+centroid1
        label[2:,:3] =  label[2:,:3]+centroid2


    np.savetxt("output.txt",label)
    


if __name__ == "__main__":

    data_dtype = "float32"
    if len(sys.argv) < 3:
        print "usage: recover_h5_to_ply xxx.h5 n(no.frame)"
        sys.exit(0)
    frame_num = int(sys.argv[2])
    file_name = sys.argv[1]
    f = h5py.File(file_name)
    data = f['data']
    label = f['label']
    print data.shape
    print label.shape
    
    recover_data(data[frame_num],label[frame_num], False, True)

