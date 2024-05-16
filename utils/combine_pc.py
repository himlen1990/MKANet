from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import sys

def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red','uint8'),('green','uint8'),('blue','uint8')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2],pc[i][3], pc[i][4], pc[i][5])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])],text=True)
    ply_out.write(filename)


if __name__ == "__main__":
    data_dtype = "float32"
    if len(sys.argv) < 3:
        print("usage: combine_pc fun.ply base.ply label.txt")
        sys.exit(0)
    pcfile1 = sys.argv[1]
    pcfile2 = sys.argv[2]
    plydata1 = PlyData.read(pcfile1)
    plydata1 = plydata1['vertex'].data
    plydata2 = PlyData.read(pcfile2)
    plydata2 = plydata2['vertex'].data
    pc_to_array1 = np.asarray(plydata1.tolist())    
    pc_to_array2 = np.asarray(plydata2.tolist())
    pc_to_array1[:,3] = 0
    pc_to_array1[:,4] = 0
    pc_to_array1[:,5] = 255
    pc_to_array2[:,3] = 255
    pc_to_array2[:,4] = 0
    pc_to_array2[:,5] = 0
    output_pc = np.r_[pc_to_array1,pc_to_array2]
    output_pc[:,:3] = output_pc[:,:3]/1000.0
    centroid = np.mean(output_pc[:,:3], axis=0)
    output_pc[:,:3] = output_pc[:,:3] - centroid
    print(centroid)
    export_ply(output_pc,"frame0000.ply")
        
