#include "../include/basketModel.h"

void BasketModel::buildBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointNormal>);
    copyCloudPointNormal(source, cloud_plane);

    findTwoPlanes(cloud_plane, _planes);

    buildLines(_planes, _lines);

    _pointsMaxMin = getMiddlePoint(source, _lines[0]);

    buildCylinder(_lines, _cylinder, _centerCylinder, _bigRadius);
    _newPoints = calculateNewPoints(_cylinder, _centerCylinder, _bigRadius, _lines[0]);

    _newPoints.push_back(_pointsMaxMin[2]);
    _PointsAxes = findPointsOnAxes(_newPoints, _centerCylinder, _smallRadius);

    computeTransformation();

    //compute 3 small cylinders coefficients
    //computeThreeCylinders(_newPoints, _lines[0], _smallRadius, _smallCylinders);
}

void BasketModel::visualizeBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                                       bool planes_flag, bool cylinder_flag, bool lines_flag)
{
    // Visualization
    pcl::visualization::PCLVisualizer vizS("PCL Result");
    vizS.addCoordinateSystem(0.1, "coordinate");
    vizS.setBackgroundColor(0.0, 0.0, 0.5);
    vizS.addPointCloud<pcl::PointNormal>(source, "source");

    if (planes_flag)
    {
        vizS.addPlane(_planes[0], "planes0");
        vizS.addPlane(_planes[1], "planes1");
    }

    if (lines_flag)
    {
        vizS.addLine(_lines[0], "line");
        vizS.addLine(_lines[1], "line_plane_a");
        vizS.addLine(_lines[2], "line_plane_b");
        vizS.addLine(_lines[3], "line_diagonal");
    }

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

    if (cylinder_flag)
    {
        vizS.addCylinder(_cylinder, "cylinder");
        //if (_smallCylinders.size() == 3)
        //{
        //vizS.addCylinder(_smallCylinders[0], "cylinder0");
        //vizS.addCylinder(_smallCylinders[1], "cylinder1");
        //vizS.addCylinder(_smallCylinders[2], "cylinder2");
        //}
    }
    vizS.spin();
}

float BasketModel::getBigRadius() { return _bigRadius; }

void BasketModel::setBigRadius(float value) { _bigRadius = value; }

float BasketModel::getSmallRadius() { return _smallRadius; }

void BasketModel::setSmallRadius(float value) { _smallRadius = value; }

Eigen::Affine3d BasketModel::getBigCylinderMatrix() { return _tBig; }

Eigen::Affine3d BasketModel::getSmallCylinderMatrix(int number)
{
    if (number == 0)
        return _tSmall0;
    else if (number == 1)
        return _tSmall1;
    else if (number == 2)
        return _tSmall2;
}

void BasketModel::copyCloudPointNormal(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
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

void BasketModel::findTwoPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes)
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

void BasketModel::buildLines(std::vector<pcl::ModelCoefficients> &planes, std::vector<pcl::ModelCoefficients> &lines)
{
    Eigen::Vector4f plane_a(planes[0].values.data());
    Eigen::Vector4f plane_b(planes[1].values.data());

    if (plane_a.w() > 0)
        plane_a = -plane_a;

    if (plane_b.w() > 0)
        plane_b = -plane_b;

    float dot_product = acos(plane_a.head<3>().dot(plane_b.head<3>())); // can be used to verify planes estimations
    if (dot_product > 2.2 || dot_product < 1.9)
        pcl::console::print_highlight("angle between planes is not consistent! value: %f\n", dot_product);

    Eigen::VectorXf line;
    pcl::planeWithPlaneIntersection(plane_a, plane_b, line);

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
    values_b[3] = plane_b.x();
    values_b[4] = plane_b.y();
    values_b[5] = plane_b.z();
    line_b.values = values_b;

    // VECTOR/LINE MIDDLE
    Eigen::Vector4f btw((plane_a + plane_b).normalized());
    pcl::ModelCoefficients line_btw; // 4
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

void BasketModel::buildCylinder(std::vector<pcl::ModelCoefficients> &lines,
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

std::vector<pcl::PointXYZ> BasketModel::calculateNewPoints(pcl::ModelCoefficients &cylinder,
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

std::vector<pcl::PointXYZ> BasketModel::getMiddlePoint(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line)
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
std::vector<pcl::PointXYZ> BasketModel::findPointsOnAxes(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float &distance)
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

void BasketModel::computeThreeCylinders(std::vector<pcl::PointXYZ> &points, pcl::ModelCoefficients line,
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

void BasketModel::computeTransformation()
{

    _centerBig = _centerCylinder.getVector3fMap();

    _centerVec.push_back(_PointsAxes[0].getVector3fMap());
    _centerVec.push_back(_PointsAxes[1].getVector3fMap());
    _centerVec.push_back(_PointsAxes[2].getVector3fMap());

    _axisDir.x() = _cylinder.values[3];
    _axisDir.y() = _cylinder.values[4];
    _axisDir.z() = _cylinder.values[5];

    _centerSmall2 = _centerVec[2];

    Eigen::Vector3f axis2 = _centerSmall2 - _centerBig;

    _axisDir.normalize();
    axis2.normalize();

    Eigen::VectorXd from_line_x, from_line_z, to_line_x, to_line_z;

    from_line_x.resize(6);
    from_line_z.resize(6);
    to_line_x.resize(6);
    to_line_z.resize(6);

    //Origin
    from_line_x << 0, 0, 0, 1, 0, 0;
    from_line_z << 0, 0, 0, 0, 0, 1;

    to_line_x.head<3>() = _centerBig.cast<double>();
    to_line_x.tail<3>() = axis2.cast<double>();

    to_line_z.head<3>() = _centerBig.cast<double>();
    to_line_z.tail<3>() = _axisDir.cast<double>();

    Eigen::Affine3d transformation;
    if (pcl::transformBetween2CoordinateSystems(from_line_x, from_line_z, to_line_x, to_line_z, transformation))
    {
        //std::cout << "Transformation matrix: \n"
        //          << transformation.matrix() << std::endl;
    }
    else
    {
        std::cout << "error computing affine transform" << std::endl;
    }

    // each cylinder is defined by its own affine transformation matrix. It is easly transformed into a pose_msg in ROS

    // big cylinder
    _tBig = transformation;

    // small cylinder 0
    _tSmall0.linear() = transformation.linear();
    _tSmall0.translation() = _centerVec[0].cast<double>();

    // small cylinder 1
    _tSmall1.linear() = transformation.linear();
    _tSmall1.translation() = _centerVec[1].cast<double>();

    // small cylinder 2
    _tSmall2.linear() = transformation.linear();
    _tSmall2.translation() = _centerVec[2].cast<double>();
}
