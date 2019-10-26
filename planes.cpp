
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "include/basketModel.h"

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

    BasketModel bm;
    bm.setBigRadius(0.4);
    bm.setSmallRadius(0.02);
    {
        pcl::ScopeTime t("Processing");
        bm.compute(source);

        Eigen::Affine3d bigMatrix = bm.getBigCylinderMatrix();

        std::cout << "big cylinder transformatation matrix: \n"
                  << bigMatrix.matrix() << std::endl;
    }
    bm.visualizeBasketModel(source, true, true, true);

    return (0);
}
