
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

public:
    void buildBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointNormal>);
        copyCloudPointNormal(source, cloud_plane);

        findTwoPlanes(cloud_plane, _planes);

        buildLines(_planes, _lines);

        _pointsMaxMin = getMiddlePoint(source, _lines[0]);

        _bigRadius = 0.2;
        buildCylinder(_lines, _cylinder, _centerCylinder, _bigRadius);
        _newPoints = calculateNewPoints(_cylinder, _centerCylinder, _bigRadius, _lines[0]);

        _newPoints.push_back(_pointsMaxMin[2]);
        _smallRadius = 0.02;
        _PointsAxes = findPointsOnAxes(_newPoints, _centerCylinder, _smallRadius);

        //compute 3 small cylinders coefficients
        //computeThreeCylinders(_newPoints, _lines[0], _smallRadius, _smallCylinders);
    }

    void visualizeBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source)
    {
        // Visualization
        pcl::visualization::PCLVisualizer vizS("PCL Result");
        vizS.addCoordinateSystem(0.1, "coordinate");
        vizS.setBackgroundColor(0.0, 0.0, 0.5);
        vizS.addPointCloud<pcl::PointNormal>(source, "source");
        //vizS.addPlane(planes[0], "planes0");
        //vizS.addPlane(planes[1], "planes1");
        //vizS.addLine(lines[0], "line");

        //vizS.addLine(lines[1], "line_plane_a");
        //vizS.addLine(lines[2], "line_plane_b");
        //vizS.addLine(lines[3], "line_diagonal");

        vizS.addSphere(_centerCylinder, 0.01, "center_cylinder");
        //vizS.addSphere(newPoints[2], 0.01, 1.0f, 0.0f, 0.0f, "line_point");
        vizS.addSphere(_newPoints[0], 0.01, 1.0f, 0.0f, 0.0f, "newPoint0");
        vizS.addSphere(_newPoints[1], 0.01, 1.0f, 0.0f, 0.0f, "newPoint1");

        vizS.addSphere(_pointsMaxMin[0], 0.01, 0.0f, 0.5f, 0.5f, "maxPoint");
        vizS.addSphere(_pointsMaxMin[1], 0.01, 0.0f, 0.5f, 0.5f, "minPoint");
        vizS.addSphere(_pointsMaxMin[2], 0.01, 0.5f, 0.5f, 0.5f, "midPoint");

        vizS.addSphere(_PointsAxes[0], 0.01, 0.8f, 0.2f, 0.5f, "PointsAxes0");
        vizS.addSphere(_PointsAxes[1], 0.01, 0.8f, 0.2f, 0.5f, "PointsAxes1");
        vizS.addSphere(_PointsAxes[2], 0.01, 0.8f, 0.2f, 0.5f, "PointsAxes2");

        vizS.addCylinder(_cylinder, "cylinder");
        //if (_smallCylinders.size() == 3)
        //{
        //vizS.addCylinder(_smallCylinders[0], "cylinder0");
        //vizS.addCylinder(_smallCylinders[1], "cylinder1");
        //vizS.addCylinder(_smallCylinders[2], "cylinder2");
        //}

        vizS.spin();
    }

    Eigen::Vector3f getBigCylinderCenter() { return _centerCylinder.getVector3fMap(); }

    std::vector<Eigen::Vector3f> getSmallCylindersCenter()
    {
        if (_PointsAxes.size() != 3)
            PCL_ERROR("wrong points vector size");

        std::vector<Eigen::Vector3f> vec;

        vec.push_back(_PointsAxes[0].getVector3fMap());
        vec.push_back(_PointsAxes[1].getVector3fMap());
        vec.push_back(_PointsAxes[2].getVector3fMap());

        return vec;
    }

    Eigen::Vector3f getCylinderAxisDirection()
    {

        Eigen::Vector3f vec;
        vec.x() = _cylinder.values[3];
        vec.y() = _cylinder.values[4];
        vec.z() = _cylinder.values[5];
        return vec;
    }

    float getBigRadius() { return _bigRadius; }

    float getSmallRadius() { return _smallRadius; }

private:
    void copyCloudPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                              pcl::PointCloud<pcl::PointNormal>::Ptr &result)
    {
        result->height = source->height;
        result->width = source->width;
        result->resize(source->size());
        for (int i = 0; i < source->size(); i++)
        {
            result->points[i].x = source->points[i].x;
            result->points[i].y = source->points[i].y;
            result->points[i].z = source->points[i].z;
            result->points[i].normal_x = source->points[i].normal_x;
            result->points[i].normal_y = source->points[i].normal_y;
            result->points[i].normal_z = source->points[i].normal_z;
            result->points[i].curvature = source->points[i].curvature;
        }
    }

    void findTwoPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointNormal> seg;
        pcl::ModelCoefficients plane;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);

        for (int i = 0; i < 2; i++)
        {
            seg.setInputCloud(cloud_plane);
            seg.segment(*inliers, plane);

            if (inliers->indices.size() == 0)
            {
                std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            }
            pcl::console::print_highlight("inliers size: %d \n", inliers->indices.size());

            pcl::ExtractIndices<pcl::PointNormal> extract;
            // Extract the inliers
            extract.setInputCloud(cloud_plane);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_plane);

            planes.push_back(plane);
        }
    }

    void buildLines(std::vector<pcl::ModelCoefficients> &planes, std::vector<pcl::ModelCoefficients> &lines)
    {
        Eigen::Vector4f plane_a(planes[0].values.data());
        //std::cout << "plane A: " << plane_a.x() << ", " << plane_a.y() << ", " << plane_a.z() << ", " << plane_a.w() << std::endl;
        Eigen::Vector4f plane_b(planes[1].values.data());
        //std::cout << "plane B: " << plane_b.x() << ", " << plane_b.y() << ", " << plane_b.z() << ", " << plane_b.w() << std::endl;

        float dot_product = acos(plane_a.head<3>().dot(plane_b.head<3>())); // can be used to verify planes estimations
        //std::cout << "dot product: " << dot_product << std::endl;

        Eigen::VectorXf line;
        pcl::planeWithPlaneIntersection(plane_a, plane_b, line, 0.1);

        pcl::ModelCoefficients line_model; // 1
        std::vector<float> values(&line[0], line.data() + line.cols() * line.rows());
        line_model.values = values;

        // LINE PLANE A
        pcl::ModelCoefficients line_a; // 2
        std::vector<float> values_a = values;
        values_a[3] = plane_a.x();
        values_a[4] = plane_a.y();
        values_a[5] = plane_a.z();
        line_a.values = values_a;

        // LINE PLANE B
        pcl::ModelCoefficients line_b; // 3
        std::vector<float> values_b = values;
        values_b[3] = -plane_b.x();
        values_b[4] = -plane_b.y();
        values_b[5] = -plane_b.z();
        line_b.values = values_b;

        // VECTOR/LINE MIDDLE
        Eigen::Vector4f btw((plane_a - plane_b).normalized()); // minus because we need to reverse the vector. Need to generalize!!!! <<<----
        pcl::ModelCoefficients line_btw;                       // 4
        std::vector<float> values_btw = values;
        values_btw[3] = btw.x();
        values_btw[4] = btw.y();
        values_btw[5] = btw.z();
        line_btw.values = values_btw;

        //retur quantities:
        lines.push_back(line_model);
        lines.push_back(line_a);
        lines.push_back(line_b);
        lines.push_back(line_btw);
    }

    void buildCylinder(std::vector<pcl::ModelCoefficients> &lines,
                       pcl::ModelCoefficients &cylinder,
                       pcl::PointXYZ &center_cylinder, float radius)
    {
        // CONSTRUCT CYLINDER
        // [point_on_axis.x point_on_axis.y point_on_axis.z axis_direction.x axis_direction.y axis_direction.z radius]
        Eigen::Vector3f point_on_axis;
        point_on_axis.x() = lines[0].values[0] - radius * lines[3].values[3];
        point_on_axis.y() = lines[0].values[1] - radius * lines[3].values[4];
        point_on_axis.z() = lines[0].values[2] - radius * lines[3].values[5];
        center_cylinder.getArray3fMap() = point_on_axis;

        std::vector<float> cylinder_values{center_cylinder.x, center_cylinder.y, center_cylinder.z,
                                           lines[0].values[3], lines[0].values[4], lines[0].values[5], 0.2};
        cylinder.values = cylinder_values;
    }

    std::vector<pcl::PointXYZ> calculateNewPoints(pcl::ModelCoefficients &cylinder,
                                                  pcl::PointXYZ &center_cylinder,
                                                  float radius_cylinder,
                                                  pcl::ModelCoefficients &line)
    {
        std::vector<pcl::PointXYZ> newPoints;
        pcl::PointXYZ new_point;

        pcl::PointXYZ line_point;
        line_point.x = line.values[0];
        line_point.y = line.values[1];
        line_point.z = line.values[2];

        // Rodrigues' rotation formula

        // angle to rotate
        float theta = (2 * M_PI) / 3.0f;

        // unit versor k
        Eigen::Vector3f k{cylinder.values[3], cylinder.values[4], cylinder.values[5]};
        k.normalize();

        // vector to rotate V
        Eigen::Vector3f V{line_point.getVector3fMap() - center_cylinder.getVector3fMap()};
        Eigen::Vector3f V_rot;

        // computation of two points, each displaced by 2pi/3
        for (int c = 0; c < 2; c++)
        {

            V_rot = V * cos(theta) + (k.cross(V)) * sin(theta) + k * (k.dot(V)) * (1 - cos(theta));
            V_rot.normalize();

            new_point.x = center_cylinder.x + radius_cylinder * V_rot.x();
            new_point.y = center_cylinder.y + radius_cylinder * V_rot.y();
            new_point.z = center_cylinder.z + radius_cylinder * V_rot.z();

            newPoints.push_back(new_point);
            V = V_rot;
        }

        return newPoints;
    }

    std::vector<pcl::PointXYZ> getMiddlePoint(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line)
    {
        pcl::PointXYZ line_point;
        line_point.x = line.values[0];
        line_point.y = line.values[1];
        line_point.z = line.values[2];

        std::vector<float> distances;
        for (int i = 0; i < cloud->size(); i++)
        {
            distances.push_back(pcl::euclideanDistance<pcl::PointXYZ, pcl::PointNormal>(line_point, cloud->points[i]));
        }

        int maxIdx = std::max_element(distances.begin(), distances.end()) - distances.begin();
        int minIdx = std::min_element(distances.begin(), distances.end()) - distances.begin();

        pcl::PointXYZ maxPoint, minPoint;

        maxPoint.x = cloud->points[maxIdx].x;
        maxPoint.y = cloud->points[maxIdx].y;
        maxPoint.z = cloud->points[maxIdx].z;

        minPoint.x = cloud->points[minIdx].x;
        minPoint.y = cloud->points[minIdx].y;
        minPoint.z = cloud->points[minIdx].z;

        //project points onto the line
        Eigen::Vector3f minP, maxP;
        minP = minPoint.getVector3fMap() - line_point.getVector3fMap();
        maxP = maxPoint.getVector3fMap() - line_point.getVector3fMap();
        Eigen::Vector3f LL{line.values[3], line.values[4], line.values[5]};

        Eigen::Vector3f minPointProjected, maxPointProjected;
        minPointProjected = minP.dot(LL) / LL.norm() * LL;
        minPointProjected.normalize();
        minPoint.getVector3fMap() = line_point.getVector3fMap() + minPointProjected * minP.norm();

        maxPointProjected = maxP.dot(LL) / LL.norm() * LL;
        maxPointProjected.normalize();
        maxPoint.getVector3fMap() = line_point.getVector3fMap() + maxPointProjected * maxP.norm();

        float finLength = pcl::euclideanDistance<pcl::PointXYZ, pcl::PointXYZ>(maxPoint, minPoint);
        pcl::console::print_highlight("Fin Length: %f meters\n", finLength);

        pcl::PointXYZ midPoint;
        midPoint.x = (maxPoint.x + minPoint.x) / 2;
        midPoint.y = (maxPoint.y + minPoint.y) / 2;
        midPoint.z = (maxPoint.z + minPoint.z) / 2;

        //modify center point
        line.values[0] = midPoint.x;
        line.values[1] = midPoint.y;
        line.values[2] = midPoint.z;

        std::vector<pcl::PointXYZ> points;
        points.push_back(maxPoint);
        points.push_back(minPoint);
        points.push_back(midPoint);
        return points;
    }
    std::vector<pcl::PointXYZ> findPointsOnAxes(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float &distance)
    {
        std::vector<pcl::PointXYZ> result;
        for (int i = 0; i < points.size(); i++)
        {
            //calculate vector from center to point and normalize it to get versor
            Eigen::Vector3f vec = points[i].getVector3fMap() - center.getVector3fMap();
            vec.normalize();

            //compute new point along versor direction
            pcl::PointXYZ newPoint;
            newPoint.getVector3fMap() = points[i].getVector3fMap() + vec * distance;

            result.push_back(newPoint);
        }
        return result;
    }

    void computeThreeCylinders(std::vector<pcl::PointXYZ> &points, pcl::ModelCoefficients line,
                               float radius, std::vector<pcl::ModelCoefficients> &smallCylinders)
    {
        pcl::ModelCoefficients cylinder;

        for (int i = 0; i < points.size(); i++)
        {
            std::vector<float> vec_values{
                points[i].x, points[i].y, points[i].z, line.values[3], line.values[4], line.values[5], radius};
            cylinder.values = vec_values;
            smallCylinders.push_back(cylinder);
        }
        return;
    }

private:
    std::vector<pcl::PointXYZ> _pointsMaxMin;
    std::vector<pcl::PointXYZ> _newPoints;
    std::vector<pcl::PointXYZ> _PointsAxes;
    std::vector<pcl::ModelCoefficients> _lines;
    std::vector<pcl::ModelCoefficients> _planes;
    std::vector<pcl::ModelCoefficients> _smallCylinders;

    pcl::ModelCoefficients _cylinder;
    pcl::PointXYZ _centerCylinder;

    float _bigRadius, _smallRadius;
};

//
//
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
    {
        pcl::ScopeTime t("Processing");
        bm.buildBasketModel(source);

        Eigen::Vector3f bigCylinderCenter = bm.getBigCylinderCenter();
        std::vector<Eigen::Vector3f> smallCylindersCenter = bm.getSmallCylindersCenter();
        Eigen::Vector3f axisDirection = bm.getCylinderAxisDirection();

        float bigRadius = bm.getBigRadius();
        float smallRadius = bm.getSmallRadius();

        pcl::console::print_highlight("big cylinder center: %f, %f, %f\n",
                                      bigCylinderCenter.x(), bigCylinderCenter.y(), bigCylinderCenter.z());
        for (int i = 0; i < smallCylindersCenter.size(); i++)
            pcl::console::print_highlight("small cylinder %d center: %f, %f, %f\n",
                                          i, smallCylindersCenter[i].x(), smallCylindersCenter[i].y(), smallCylindersCenter[i].z());
        pcl::console::print_highlight("axis cylinder direction: %f, %f, %f\n",
                                      axisDirection.x(), axisDirection.y(), axisDirection.z());
        pcl::console::print_highlight("big cylinder radius %f, small cylinder radius %f\n", bigRadius, smallRadius);
    }
    //bm.visualizeBasketModel(source);

    return (0);
}