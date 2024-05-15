from plyfile import (PlyData, PlyElement, PlyParseError, PlyProperty)
import numpy as np
import random
import h5py
import glob
import os
import math
import time

dataset_dir = "./dataset/train_data/"
function_category = 1

def label_normalize(pointcloud,label,has_function_part=True):
    if has_function_part:
        function_part = pointcloud[(pointcloud[:,5]==255).nonzero()][:,:3]
    else:
        function_part = pointcloud[:,:3] # using the whole object
    centroid = np.mean(function_part, axis=0)            
    transform =  label[:,:3]-centroid
    norm_label = np.c_[transform,label[:,3:]]
    return norm_label


def load_point_cloud_and_label(obj_dir, obj_list, label_list):
    for obj_num in glob.glob(os.path.join(obj_dir,'*')):
        for plyfile in glob.glob(os.path.join(obj_num,'*.ply')):
            print("processing file: ",plyfile)
            plydata = PlyData.read(plyfile)
            pc = plydata['vertex'].data
            pc_to_array = np.asarray(pc.tolist())
            obj_list.append(pc_to_array)
            txtfile = plyfile[:-4] + '.txt'
            labeldata = np.loadtxt(txtfile)
            labeldata = np.reshape(labeldata, (-1,7))
            norm_label = label_normalize(pc_to_array,labeldata)
            label_list.append(norm_label)

def generate_scence(obj1_dir, obj2l_dir, obj2r_dir, num_scenes=100):

    obj1 = []
    label1 = []
    obj2l = []
    label2l = []
    obj2r = []
    label2r = []
    num_points = 2400
    load_point_cloud_and_label(obj1_dir, obj1, label1)
    load_point_cloud_and_label(obj2l_dir, obj2l, label2l)
    load_point_cloud_and_label(obj2r_dir, obj2r, label2r)
    
    generated_scene = []
    generated_scene_label = []
    count_left = 0
    count_right = 0
    random.seed(time.time())
    for i in range(num_scenes):
        side_flag = False # false-> right, true-> left

        cloud_idx1 = random.sample([*range(len(obj1))], 1) 

        jitter_x1 = random.uniform(0.2,1.0)
        jitter_y1 = random.uniform(-0.7,0.7)
        jitter_z1 = random.uniform(0.5,1.0)
        jittered_cloud1 = obj1[cloud_idx1[0]].copy() 

        jittered_cloud1[:,0] = jittered_cloud1[:,0] + jitter_x1
        jittered_cloud1[:,1] = jittered_cloud1[:,1] + jitter_y1
        jittered_cloud1[:,2] = jittered_cloud1[:,2] + jitter_z1

        jitter_x2 = random.uniform(0,1.0)
        jitter_y2 = random.uniform(-0.5,0.5)+jitter_y1
        jitter_z2 = random.uniform(0,1.0)

        if jitter_y2 > jitter_y1:                        
            side_flag = True
            cloud_idx2 = random.sample([*range(len(obj2l))], 1) 
            count_left = count_left+1
        else: 
            cloud_idx2 = random.sample([*range(len(obj2r))], 1) 
            count_right= count_right+1

        dis = math.sqrt((jitter_x2-jitter_x1)**2 +
                        (jitter_y2-jitter_y1)**2 +
                        (jitter_z2-jitter_z1)**2)
        while(dis<0.2):
            print("dis less than 0.2, dis= ", dis)
            jitter_x2 = random.uniform(0,1.0)
            jitter_z2 = random.uniform(0,1.0)
            dis = math.sqrt((jitter_x2-jitter_x1)**2 +
                            (jitter_y2-jitter_y1)**2 +
                            (jitter_z2-jitter_z1)**2)
            
        if side_flag:
            jittered_cloud2 = obj2l[cloud_idx2[0]].copy()
        else:
            jittered_cloud2 = obj2r[cloud_idx2[0]].copy()

        jittered_cloud2[:,0] = jittered_cloud2[:,0] + jitter_x2
        jittered_cloud2[:,1] = jittered_cloud2[:,1] + jitter_y2
        jittered_cloud2[:,2] = jittered_cloud2[:,2] + jitter_z2


        #background part
        jittered_cloud1[jittered_cloud1[:,3]>0, 4] = 0 #turn red color points to label 0
        jittered_cloud2[jittered_cloud2[:,3]>0, 4] = 0

        #function part
        jittered_cloud1[jittered_cloud1[:,5]>0, 4] = function_category #turn blue color points to label 1
        jittered_cloud2[jittered_cloud2[:,5]>0, 4] = function_category

        #add activation, 1 is activated
        jittered_cloud1[:,3] = 1
        jittered_cloud2[:,3] = 0

        scene = np.concatenate([jittered_cloud1,jittered_cloud2])[:,:5]

        sample_points_idx = random.sample([*range(scene.shape[0])], int(scene.shape[0]*1/2))
        sampled_pc = scene[sample_points_idx,:]
        if sampled_pc.shape[0] > num_points:
                num_drop = sampled_pc.shape[0] - num_points
                drop_idx = random.sample([*range(sampled_pc.shape[0])], num_drop) 
                sampled_pc[drop_idx, 4] = -1
                reshape_pc = sampled_pc[sampled_pc[:,4] > -1]
        elif sampled_pc.shape[0] < num_points: # pad with zeros
                num_padding = num_points - sampled_pc.shape[0] 
                padding = np.zeros((num_padding, sampled_pc.shape[1]), dtype = sampled_pc.dtype)
                reshape_pc = np.append(sampled_pc,padding, axis=0)                
        generated_scene.append(reshape_pc)

        if side_flag:
            obj2_lable = label2l[cloud_idx2[0]]
        else:
            obj2_lable = label2r[cloud_idx2[0]]

        scene_label = np.r_[label1[cloud_idx1[0]],obj2_lable]        
        generated_scene_label.append(scene_label)

    print("left: ", count_left)
    print("right: ", count_right)
    return np.array(generated_scene),np.array(generated_scene_label)

if __name__ == "__main__":

    pair_list = [["bottle","cup"]]
    #pair_list = [["cup","bowl"]]

    for pair in pair_list:
        obj1_dir = dataset_dir + pair[0] + "/pour/"
        obj2l_dir = dataset_dir + pair[1] + "/poured_l/"
        obj2r_dir = dataset_dir + pair[1] + "/poured_r/"
        file_name = "./scene_" + pair[0] + pair[1] + ".h5"


        data_arr,label_arr = generate_scence(obj1_dir,obj2l_dir,obj2r_dir, 200)    

        data_dtype = "float32"

        h5_fout=h5py.File(file_name,'w')
        h5_fout.create_dataset('data', data=data_arr,
                               compression='gzip', compression_opts=4,
                               dtype=data_dtype)
        h5_fout.create_dataset('label', data=label_arr,
                               compression='gzip', compression_opts=4,
                               dtype=data_dtype)
        h5_fout.close()

        
        f = h5py.File(file_name)
        data = f['data']

