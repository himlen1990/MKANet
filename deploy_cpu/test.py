import tensorflow as tf
import numpy as np
import h5py
import os 
import random
import sys
import matplotlib.pyplot as plt
from plyfile import PlyData,PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet_utils'))
sys.path.append(os.path.join(BASE_DIR, 'pointnet_utils/render_ball'))
import mkanet as model

def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red','uint8'),('green','uint8'),('blue','uint8')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2],pc[i][3], pc[i][4], pc[i][5])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])],text=True)
    ply_out.write(filename)

def load_dataset(file_name):
    f = h5py.File(file_name)
    data = f['data'][:,:,:4]
    label = f['data'][:,:,4]
    return data, label

def load_ply(plyfile):
    plydata = PlyData.read(plyfile)
    pc = plydata['vertex'].data
    pc_to_array = np.asarray(pc.tolist())[:,:4]
    return pc_to_array


def predict(data):
    feed_dict = {pointclouds_pl: [data],
                 is_training_pl: False}
    logits, reglabel = sess.run([clspred,regpred],feed_dict=feed_dict)
    return np.argmax(logits,2), reglabel[0]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: test.py file_name.ply")
        sys.exit(0)
    file_name = sys.argv[1]
    pointcloud = load_ply(file_name)
    pointclouds_pl, clslabels_pl, reglabels_pl = model.placeholder_inputs(1,2400)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    num_class = 2
    sess = tf.Session()
    clspred, regpred, end_points = model.get_model(pointclouds_pl,is_training_pl,num_class)
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess, "log/model.ckpt")
    segp, traj_label = predict(pointcloud)

    #object state recovery
    padding = np.zeros((pointcloud.shape[0],1))
    output_pc = np.c_[pointcloud,segp[0],padding]
    obj1 = output_pc[(output_pc[:,3]==1).nonzero()]
    obj2 = output_pc[(output_pc[:,3]==0).nonzero()]
    obj1[:,3] = 255
    obj1_function_part = obj1[(obj1[:,4] > 0).nonzero()][:,0:3]
    obj2_function_part = obj2[(obj2[:,4] > 0).nonzero()][:,0:3]
    export_pc = np.r_[obj1,obj2]
    centroid1 = np.mean(obj1_function_part, axis=0)
    centroid2 = np.mean(obj2_function_part, axis=0)    
    traj_label[:1,:3] =  traj_label[:1,:3]+centroid1
    traj_label[1:,:3] =  traj_label[1:,:3]+centroid2
    export_ply(export_pc,"results/predicted.ply")
    np.savetxt("results/predicted.txt",traj_label)
