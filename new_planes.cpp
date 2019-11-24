
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "include/newBasketModel.h"

int main(int argc, char **argv)
{
    // Point clouds
    pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);

    // Get input object and scene
    if (argc != 2)
    {
        pcl::console::print_error("error", argv[0]);
        return (1);
    }

    // Load object and scene
    pcl::console::print_highlight("Loading point cloud...\n");
    if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[1], *source) < 0)
    {
        pcl::console::print_error("Error loading file!\n");
        return (1);
    }

    Eigen::Vector3f basketCenter;
    basketCenter.x() = 0.0529266;
    basketCenter.y() = 0.28118;
    basketCenter.z() = 0.0158598;

    Eigen::Vector3f basketAxisDir;
    basketAxisDir.x() = -0.056495;
    basketAxisDir.y() = -0.686225;
    basketAxisDir.z() = -0.559771;

    DrumModel bm;
    bm.setBasketCenter(basketCenter);
    bm.setBasketAxis(basketAxisDir);
    bm.compute(source);

    bm.visualizeBasketModel(source, false, true, true);

    return (0);
}
