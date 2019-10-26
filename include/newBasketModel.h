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

class BasketModel2
{
public:
    Eigen::Vector3f _basketCenter;
    Eigen::Vector3f _basketAxisVector;

    pcl::PointXYZ _maxFinPoint, _minFinPoint, _centerFinPoint, _centerFinProjected;
    pcl::PointXYZ _centerFin2Point, _centerFin3Point;
    std::vector<pcl::PointXYZ> _finsCenter;

    std::vector<pcl::ModelCoefficients> _planes;
    pcl::ModelCoefficients _line, _lineBasket, _cylinder;

    Eigen::Affine3d _tBig, _tSmall0, _tSmall1, _tSmall2;

    float _finLength, _finHeight;

public:
    void setBasketCenter(Eigen::Vector3f basketCenter);
    void setBasketAxis(Eigen::Vector3f basketAxisDir);

    Eigen::Affine3d getBigMatrix();
    Eigen::Affine3d getSmallgMatrix0();
    Eigen::Affine3d getSmallgMatrix1();
    Eigen::Affine3d getSmallgMatrix2();

    void visualizeBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                              bool planes_flag, bool cylinder_flag, bool lines_flag);

    void compute(pcl::PointCloud<pcl::PointNormal>::Ptr &input);

    void findPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes);

    void estimateIntersactionLine(std::vector<pcl::ModelCoefficients> &planes, pcl::ModelCoefficients &line_model);

    void getPointsOnLine(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line,
                         pcl::PointXYZ &maxFinPoint, pcl::PointXYZ &minFinPoint, pcl::PointXYZ &centerFinPoint);

    void checkIfParallel(pcl::ModelCoefficients &line, pcl::ModelCoefficients &axisBasket);

    void buildLineModelCoefficient(Eigen::Vector3f &point, Eigen::Vector3f &axis, pcl::ModelCoefficients &_lineBasket);

    pcl::PointXYZ projection(pcl::PointXYZ &point, pcl::ModelCoefficients &line);

    void calculateNewPoints(pcl::ModelCoefficients &cylinder, pcl::PointXYZ &centerCylinder,
                            pcl::PointXYZ &centerFin1, pcl::PointXYZ &centerFin2, pcl::PointXYZ &centerFin3);

    std::vector<pcl::PointXYZ> movePointsToFinsCenter(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float distance);

    pcl::ModelCoefficients buildModelCoefficientCylinder(pcl::PointXYZ pointFIn, pcl::PointXYZ pointFinProj, Eigen::Vector3f axis);

    float calculateFinHeight(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::ModelCoefficients &line);

    void computeTransformation();
};