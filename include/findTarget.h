/*
Robust pose estimation of rigid objects
how to find the alignment pose of a rigid object in a scene with clutter and occlusions.

http://pointclouds.org/documentation/tutorials/alignment_prerejective.php#alignment-prerejective

*/
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/registration/icp.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

class FindTarget
{
public:
    PointCloudT::Ptr object;
    PointCloudT::Ptr object_aligned;
    PointCloudT::Ptr scene;
    FeatureCloudT::Ptr object_features;
    FeatureCloudT::Ptr scene_features;
    PointCloudT::Ptr object_icp;

public:
    FindTarget() : object(new PointCloudT),
                   object_aligned(new PointCloudT),
                   object_icp(new PointCloudT),
                   scene(new PointCloudT),
                   object_features(new FeatureCloudT),
                   scene_features(new FeatureCloudT)
    {
    }

    bool compute();

    void visualize(bool spinFlag = true);
};