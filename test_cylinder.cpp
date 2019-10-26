
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>

#include <pcl/visualization/pcl_visualizer.h>

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
    basketCenter.x() = -0.0359036;
    basketCenter.y() = -0.29824;
    basketCenter.z() = 0.494309;

    Eigen::Vector3f basketAxisDir;
    basketAxisDir.x() = 0.0205354;
    basketAxisDir.y() = 0.631626;
    basketAxisDir.z() = -0.621389;

    pcl::ModelCoefficients line;
    std::vector<float> values{basketCenter.x(), basketCenter.y(), basketCenter.z(),
                              basketAxisDir.x(), basketAxisDir.y(), basketAxisDir.z()};
    line.values = values;

    pcl::ModelCoefficients cylinder;
    std::vector<float> values2{basketCenter.x(), basketCenter.y(), basketCenter.z(),
                               basketAxisDir.x(), basketAxisDir.y(), basketAxisDir.z(), 0.16};
    cylinder.values = values2;

    pcl::visualization::PCLVisualizer viz("PCL VIZ");
    viz.setBackgroundColor(0.0, 0.0, 0.5);
    viz.addCoordinateSystem(0.1, "coordinate");
    viz.addPointCloud<pcl::PointNormal>(source, "source");
    viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0f, 0.0f, 0.0f, "source");

    viz.addLine(line, "line");
    viz.addCylinder(cylinder, "cylinder");

    viz.spin();

    return (0);
}

// Basket5
//    Eigen::Vector3f basketCenter;
//    basketCenter.x() = -0.0359036;
//    basketCenter.y() = -0.29824;
//    basketCenter.z() = 0.494309;
//
//    Eigen::Vector3f basketAxisDir;
//    basketAxisDir.x() = 0.0205354;
//    basketAxisDir.y() = 0.631626;
//    basketAxisDir.z() = -0.621389;