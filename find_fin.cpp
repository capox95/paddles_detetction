
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>

#include "include/findTarget.h"

typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

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
    ft.compute();
  }
  ft.visualize();

  return (0);
}
