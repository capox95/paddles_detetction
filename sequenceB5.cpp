#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "include/findTarget.h"
#include "include/newBasketModel.h"

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

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

    // for basket5
    Eigen::Vector3f basketCenter;
    basketCenter.x() = -0.0359036;
    basketCenter.y() = -0.29824;
    basketCenter.z() = 0.494309;

    Eigen::Vector3f basketAxisDir;
    basketAxisDir.x() = 0.0205354;
    basketAxisDir.y() = 0.631626;
    basketAxisDir.z() = -0.621389;

    FindTarget ft;
    ft.object = object;
    ft.scene = scene;
    bool success = ft.compute();
    if (!success)
        return -1;
    ft.visualize(false);

    BasketModel2 bm;
    bm.setBasketCenter(basketCenter);
    bm.setBasketAxis(basketAxisDir);
    bm.compute(ft.object_icp);

    Eigen::Affine3d tBig = bm.getBigMatrix();
    Eigen::Affine3d tS0 = bm.getSmallgMatrix0();
    Eigen::Affine3d tS1 = bm.getSmallgMatrix1();
    Eigen::Affine3d tS2 = bm.getSmallgMatrix2();

    std::cout << "bigMatrix: \n"
              << tBig.matrix() << "\n"
              << std::endl;
    std::cout << "smallMatrix0: \n"
              << tS0.matrix() << "\n"
              << std::endl;
    std::cout << "smallMatrix2: \n"
              << tS1.matrix() << "\n"
              << std::endl;
    std::cout << "smallMatrix3:  \n"
              << tS2.matrix() << "\n"
              << std::endl;

    bm.visualizeBasketModel(scene, false, true, true);

    return (0);
}
