#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/common/transforms.h>

#include <time.h>
#include <fstream>
#include <math.h>
pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudinor (new pcl::PointCloud<pcl::PointXYZINormal>);  

using namespace std;     
         
std::vector <Eigen::VectorXf> read_labels(char* file_path)
{
  std::ifstream file;
  std::vector <Eigen::VectorXf> poses;
  file.open(file_path,std::ios_base::in);
  if (file.fail()) {
    std::cout << "label file does not exist " << std::endl;
    return poses;
  }
  std::string line;
  while (std::getline(file, line))
    {
      std::stringstream ss(line);
      std::vector <float> element;
      while (getline(ss,line,' '))
		{
		  element.push_back(std::atof(line.c_str()));	  
		}
	  //std vector to eigen vector;
	  Eigen::VectorXf pose = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(element.data(), element.size());  
      poses.push_back(pose);
    }
  return poses;
}

// t->[0,1] t=0 -> return qa, t=1 -> return qb
Eigen::Quaternionf slerp(Eigen::Quaternionf qa, Eigen::Quaternionf qb, float t) 
{
  // quaternion to return
  Eigen::Quaternionf qm;
  // Calculate angle between them.
  float cosHalfTheta = qa.w() * qb.w() + qa.x() * qb.x() + qa.y() * qb.y() + qa.z() * qb.z();
  // if qa=qb or qa=-qb then theta = 0 and we can return qa
  if (abs(cosHalfTheta) >= 1.0)
	{
	  qm.w() = qa.w(); 
	  qm.x() = qa.x(); 
	  qm.y() = qa.y(); 
	  qm.z() = qa.z();
	  return qm;
	}
  // Calculate temporary values.
  float halfTheta = acos(cosHalfTheta);
  float sinHalfTheta = sqrt(1.0 - cosHalfTheta*cosHalfTheta);
  // if theta = 180 degrees then result is not fully defined
  // we could rotate around any axis normal to qa or qb
  if (fabs(sinHalfTheta) < 0.001)
	{ // fabs is floating point absolute
	  qm.w() = (qa.w() * 0.5 + qb.w() * 0.5);
	  qm.x() = (qa.x() * 0.5 + qb.x() * 0.5);
	  qm.y() = (qa.y() * 0.5 + qb.y() * 0.5);
	  qm.z() = (qa.z() * 0.5 + qb.z() * 0.5);
	  return qm;
	}
  double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
  double ratioB = sin(t * halfTheta) / sinHalfTheta; 
  //calculate Quaternion.
  qm.w() = (qa.w() * ratioA + qb.w() * ratioB);
  qm.x() = (qa.x() * ratioA + qb.x() * ratioB);
  qm.y() = (qa.y() * ratioA + qb.y() * ratioB);
  qm.z() = (qa.z() * ratioA + qb.z() * ratioB);
  return qm;
}

void trans_cloud(char* cloud_path,char* file_path){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object1 (new pcl::PointCloud<pcl::PointXYZRGB>);   
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr object1_inter (new pcl::PointCloud<pcl::PointXYZRGB>);   
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show (new pcl::PointCloud<pcl::PointXYZRGB>);  
  pcl::PLYReader Reader;
  Reader.read(cloud_path, *cloud);

  for(int i=0 ; i<cloud->points.size(); i++)
	{
	  if(cloud->points[i].r == 255)
		{
		  pcl::PointXYZRGB point;
		  point.x = cloud->points[i].x;
		  point.y = cloud->points[i].y;
		  point.z = cloud->points[i].z;
		  point.r = cloud->points[i].r;// = 0;
		  point.g = cloud->points[i].g;
		  point.b = cloud->points[i].b;

		  object1->points.push_back(point);
		}
	}

  for(int i=0 ; i<cloud->points.size(); i++)
	{
	  if(cloud->points[i].r == 255)
		{
		  pcl::PointXYZRGB point;
		  point.x = cloud->points[i].x;
		  point.y = cloud->points[i].y;
		  point.z = cloud->points[i].z;
		  point.r = 0;
		  point.g = 255;
		  point.b = 0;

		  object1_inter->points.push_back(point);
		}
	}


  pcl::copyPointCloud(*cloud, *cloud_show);
  //handle label
  std::vector <Eigen::VectorXf> label = read_labels(file_path);  
  cout<<"original num of labels: "<<label.size()<<endl;

  //interpolation

  Eigen::Quaternionf qa;
  qa.w() = label[0][3];
  qa.x() = label[0][4];
  qa.y() = label[0][5];
  qa.z() = label[0][6];

  Eigen::Quaternionf qb;
  qb.w() = label[1][3];
  qb.x() = label[1][4];
  qb.y() = label[1][5];
  qb.z() = label[1][6];

  int num_interp = 5;
  for(int i=1; i< num_interp; i++)
	{
	  float ratio = float(i)/num_interp;
	  Eigen::VectorXf label_interp = (label[1]-label[0])*ratio + label[0];
	  Eigen::Quaternionf qm = slerp(qa,qb,ratio);
	  label_interp[3]=qm.w();
	  label_interp[4]=qm.x();
	  label_interp[5]=qm.y();
	  label_interp[6]=qm.z();
	  label.push_back(label_interp);
	}

  Eigen::Quaternionf qa2;
  qa2.w() = label[2][3];
  qa2.x() = label[2][4];
  qa2.y() = label[2][5];
  qa2.z() = label[2][6];

  Eigen::Quaternionf qb2;
  qb2.w() = label[1][3];
  qb2.x() = label[1][4];
  qb2.y() = label[1][5];
  qb2.z() = label[1][6];

  int num_interp2 = 0;
  for(int i=1; i< num_interp2; i++)
	{
	  float ratio = float(i)/num_interp2;
	  Eigen::VectorXf label_interp = (label[1]-label[2])*ratio + label[2];

	  //if use Quaternion interpolation
	  Eigen::Quaternionf qm = slerp(qa2,qb2,ratio);
	  label_interp[3]=qm.w();
	  label_interp[4]=qm.x();
	  label_interp[5]=qm.y();
	  label_interp[6]=qm.z();
	  label.push_back(label_interp);
	}

  std::string txt_path = "traj_label_after_interpolation.txt";
  std::ofstream file;
  file.open(txt_path.c_str());
  for(int i=0; i<label.size(); i++)
	{
	  file<<label[i][0]<<" "<<label[i][1]<<" "<<label[i][2]<<" "<<label[i][3]<<" "<<label[i][4]<<" "<<label[i][5]<<" "<<label[i][6]<<std::endl;
	}
  file.close();
  std::vector <Eigen::Matrix4f> frames;
  for(int i=0; i<label.size(); i++)
	{
	  Eigen::Quaternionf q;
	  q.w() = label[i][3];
	  q.x() = label[i][4];
	  q.y() = label[i][5];
	  q.z() = label[i][6];

	  Eigen::Matrix3f RM = q.normalized().toRotationMatrix();  
	  Eigen::Matrix4f frame = Eigen::Matrix4f::Identity();

	  frame.block<3,3>(0,0) = RM;

	  frame(0,3) = label[i][0];
	  frame(1,3) = label[i][1];
	  frame(2,3) = label[i][2];
	  frames.push_back(frame);
	}

  for(int i=1; i<3; i++)
	{
	  Eigen::Matrix4f trans =  frames[i]*frames[0].inverse();
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
	  pcl::transformPointCloud (*object1, *cloud_trans, trans);
	  *cloud_show += *cloud_trans;
	}

  for(int i=3; i<frames.size(); i++)
	{
	  Eigen::Matrix4f trans =  frames[i]*frames[0].inverse();
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
	  pcl::transformPointCloud (*object1_inter, *cloud_trans, trans);
	  *cloud_show += *cloud_trans;
	}


  pcl::PLYWriter ply_saver;
  ply_saver.write("./result.ply",*cloud_show);


  pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> visColorSource(cloud_show);
  viewer.setBackgroundColor (1.0, 1.0, 1.0);

  viewer.addPointCloud<pcl::PointXYZRGB> (cloud_show, visColorSource, "source cloud");
  viewer.initCameraParameters ();

  while (!viewer.wasStopped ())
    {
      viewer.spinOnce();
    }

}

int main (int argc, char** argv)
{
  if (argc<3)
    {
      cout<<"usage: ply_data_process pointcloud_file label_file"<<endl;
      return -1;
    }
  
  trans_cloud(argv[1],argv[2]);
  return 0;
}
