
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/common/intersections.h>
#include <pcl/common/centroid.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

class BasketModel
{

private:
    std::vector<pcl::PointXYZ> _pointsMaxMin;
    std::vector<pcl::PointXYZ> _newPoints;
    std::vector<pcl::PointXYZ> _PointsAxes;
    std::vector<pcl::ModelCoefficients> _lines;
    std::vector<pcl::ModelCoefficients> _planes;
    std::vector<pcl::ModelCoefficients> _smallCylinders;

    pcl::ModelCoefficients _cylinder;
    pcl::PointXYZ _centerCylinder;

    Eigen::Vector3f _centerBig, _axisDir, _centerSmall0, _centerSmall1, _centerSmall2;
    std::vector<Eigen::Vector3f> _centerVec;
    Eigen::Affine3d _tBig, _tSmall0, _tSmall1, _tSmall2;

    float _bigRadius,
        _smallRadius;


public:
    BasketModel()
    {
        _bigRadius = 0.2;
        _smallRadius = 0.02;
    }

    void compute(pcl::PointCloud<pcl::PointNormal>::Ptr &source);

    void visualizeBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                              bool planes_flag = false, bool cylinder_flag = false, bool lines_flag = false);

    float getBigRadius();

    void setBigRadius(float value);

    float getSmallRadius();

    void setSmallRadius(float value);

    Eigen::Affine3d getBigCylinderMatrix();

    Eigen::Affine3d getSmallCylinderMatrix(int number);

private:
    void copyCloudPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                              pcl::PointCloud<pcl::PointNormal>::Ptr &result);

    void findTwoPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes);

    void buildLines(std::vector<pcl::ModelCoefficients> &planes, std::vector<pcl::ModelCoefficients> &lines);

    void buildCylinder(std::vector<pcl::ModelCoefficients> &lines,
                       pcl::ModelCoefficients &cylinder,
                       pcl::PointXYZ &center_cylinder, float radius);

    std::vector<pcl::PointXYZ> calculateNewPoints(pcl::ModelCoefficients &cylinder,
                                                  pcl::PointXYZ &center_cylinder,
                                                  float radius_cylinder,
                                                  pcl::ModelCoefficients &line);

    std::vector<pcl::PointXYZ> getMiddlePoint(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line);

    std::vector<pcl::PointXYZ> findPointsOnAxes(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float &distance);

    void computeThreeCylinders(std::vector<pcl::PointXYZ> &points, pcl::ModelCoefficients line,
                               float radius, std::vector<pcl::ModelCoefficients> &smallCylinders);

    void computeTransformation();


};