#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "include/findTarget.h"
#include "include/basketModel.h"

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

    pcl::PointCloud<pcl::PointNormal>::Ptr result(new pcl::PointCloud<pcl::PointNormal>);
    FindTarget ft;
    ft.object = object;
    ft.scene = scene;
    ft.compute();

    /* // ft.object_icp is PointNormal instead the class expected a PointXYZ !1!!!!!
    BasketModel bm;
    bm.setBigRadius(0.3);
    bm.setSmallRadius(0.02);
    bm.buildBasketModel(ft.object_icp);

    Eigen::Affine3d bigMatrix = bm.getBigCylinderMatrix();
    Eigen::Affine3d smallMatrix0 = bm.getSmallCylinderMatrix(0);
    Eigen::Affine3d smallMatrix2 = bm.getSmallCylinderMatrix(1);
    Eigen::Affine3d smallMatrix3 = bm.getSmallCylinderMatrix(2);

    std::cout << "bigMatrix: \n"
              << bigMatrix.matrix() << std::endl;
    std::cout << "smallMatrix0: \n"
              << smallMatrix0.matrix() << std::endl;
    std::cout << "smallMatrix2: \n"
              << smallMatrix2.matrix() << std::endl;
    std::cout << "smallMatrix3:  \n"
              << smallMatrix3.matrix() << std::endl;

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud(ft.object_aligned, ColorHandlerT(ft.object_aligned, 0.0, 0.0, 255.0), "object_aligned");

    bm.visualizeBasketModel(scene);

    visu.spin();
*/
    return (0);
}
