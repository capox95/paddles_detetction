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
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

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

public:
  FindTarget() : object(new PointCloudT),
                 object_aligned(new PointCloudT),
                 scene(new PointCloudT),
                 object_features(new FeatureCloudT),
                 scene_features(new FeatureCloudT)
  {
  }

  void findFin(pcl::PointCloud<pcl::PointXYZ>::Ptr &result)
  {
    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    const float leaf = 0.005f;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(object);
    grid.filter(*object);
    grid.setInputCloud(scene);
    grid.filter(*scene);

    // Estimate normals for scene
    pcl::console::print_highlight("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest;
    nest.setNumberOfThreads(4);
    nest.setRadiusSearch(0.01);
    nest.setInputCloud(scene);
    nest.compute(*scene);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch(0.025);
    fest.setInputCloud(object);
    fest.setInputNormals(object);
    fest.compute(*object_features);
    fest.setInputCloud(scene);
    fest.setInputNormals(scene);
    fest.compute(*scene_features);

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
    align.setInputSource(object);
    align.setSourceFeatures(object_features);
    align.setInputTarget(scene);
    align.setTargetFeatures(scene_features);
    align.setMaximumIterations(50000);               // Number of RANSAC iterations
    align.setNumberOfSamples(3);                     // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(5);            // Number of nearest features to use
    align.setSimilarityThreshold(0.95f);             // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(2.5f * leaf); // Inlier threshold
    align.setInlierFraction(0.25f);                  // Required inlier fraction for accepting a pose hypothesis
    {
      pcl::ScopeTime t("Alignment");
      align.align(*object_aligned);
    }

    if (align.hasConverged())
    {
      // Print results
      printf("\n");
      Eigen::Matrix4f transformation = align.getFinalTransformation();
      pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1), transformation(0, 2));
      pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1), transformation(1, 2));
      pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1), transformation(2, 2));
      pcl::console::print_info("\n");
      pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3), transformation(2, 3));
      pcl::console::print_info("\n");
      pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), object->size());

      std::vector<int> inliers = align.getInliers();
      std::cout << object_aligned->size() << std::endl;

      result->width = inliers.size();
      result->height = 1;
      result->resize(result->width * result->height);
      for (int i = 0; i < inliers.size(); i++)
      {
        int id = inliers[i];
        result->points[i].x = object_aligned->points[id].x;
        result->points[i].y = object_aligned->points[id].y;
        result->points[i].z = object_aligned->points[id].z;
      }
    }
    else
    {
      pcl::console::print_error("Alignment failed!\n");
      return;
    }
  }
};

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
  // Point clouds
  PointCloudT::Ptr object(new PointCloudT);
  PointCloudT::Ptr scene(new PointCloudT);

  // Get input object and scene
  if (argc != 3)
  {
    pcl::console::print_error("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    return (1);
  }

  // Load object and scene
  pcl::console::print_highlight("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[1], *object) < 0 ||
      pcl::io::loadPCDFile<pcl::PointNormal>(argv[2], *scene) < 0)
  {
    pcl::console::print_error("Error loading object/scene file!\n");
    return (1);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
  FindTarget ft;
  ft.object = object;
  ft.scene = scene;
  {
    pcl::ScopeTime t("Processing");
    ft.findFin(result);
  }

  pcl::visualization::PCLVisualizer visuR("Result");
  visuR.addPointCloud(result, "result");

  // Show alignment
  pcl::visualization::PCLVisualizer visu("Alignment");
  visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene");
  visu.addPointCloud(ft.object_aligned, ColorHandlerT(ft.object_aligned, 0.0, 0.0, 255.0), "object_aligned");
  visu.spin();

  return (0);
}
