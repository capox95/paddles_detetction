#include "../include/newBasketModel.h"

void DrumModel::setBasketCenter(Eigen::Vector3f basketCenter) { _basketCenter = basketCenter; };
void DrumModel::setBasketAxis(Eigen::Vector3f basketAxisDir) { _basketAxisVector = basketAxisDir; };

Eigen::Affine3d DrumModel::getBigMatrix() { return _tBig; }
Eigen::Affine3d DrumModel::getSmallgMatrix0() { return _tSmall0; }
Eigen::Affine3d DrumModel::getSmallgMatrix1() { return _tSmall1; }
Eigen::Affine3d DrumModel::getSmallgMatrix2() { return _tSmall2; }

float DrumModel::getFinLength() { return _finLength; }
float DrumModel::getFinHeight() { return _finHeight; } // actually is half the height

void DrumModel::visualizeBasketModel(pcl::PointCloud<pcl::PointNormal>::Ptr &source,
                                     bool planes_flag, bool cylinder_flag, bool lines_flag)
{
    // Visualization
    pcl::visualization::PCLVisualizer vizS("PCL");
    vizS.addCoordinateSystem(0.1, "coordinate");
    vizS.setBackgroundColor(1.0, 1.0, 1.0);
    vizS.addPointCloud<pcl::PointNormal>(source, "source");
    vizS.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.7, 0.0, "source");

    if (planes_flag)
    {
        vizS.addPlane(_planes[0], "planes0");
        vizS.addPlane(_planes[1], "planes1");
    }

    if (lines_flag)
    {
        vizS.addLine(_line, "line");
        vizS.addLine(_lineBasket, "_axisBasket");
    }
    vizS.addSphere(_maxFinPoint, 0.01, 1.0f, 0.0f, 0.0f, "_maxFinPoint");
    vizS.addSphere(_minFinPoint, 0.01, 1.0f, 0.0f, 0.0f, "_minFinPoint");
    vizS.addSphere(_centerFinProjected, 0.01, 1.0f, 1.0f, 0.0f, "_projectedPoint");

    vizS.addSphere(_centerFinPoint, 0.01, 1.0f, 1.0f, 0.0f, "_centerFinPoint");
    vizS.addSphere(_centerFin2Point, 0.01, 1.0f, 1.0f, 0.0f, "_centerFin2Point");
    vizS.addSphere(_centerFin3Point, 0.01, 1.0f, 1.0f, 0.0f, "_centerFin3Point");

    vizS.addSphere(_finsCenter[0], 0.01, 0.0f, 0.0f, 1.0f, "_finsCenter0");
    vizS.addSphere(_finsCenter[1], 0.01, 0.0f, 0.0f, 1.0f, "_finsCenter1");
    vizS.addSphere(_finsCenter[2], 0.01, 0.0f, 0.0f, 1.0f, "_finsCenter2");

    if (cylinder_flag)
        vizS.addCylinder(_cylinder, "cylinder");

    vizS.spin();
}

void DrumModel::compute(pcl::PointCloud<pcl::PointNormal>::Ptr &input)
{
    findPlanes(input, _planes);
    estimateIntersactionLine(_planes, _line);
    getPointsOnLine(input, _line, _maxFinPoint, _minFinPoint, _centerFinPoint);
    calculateFinHeight(input, _line);

    buildLineModelCoefficient(_basketCenter, _basketAxisVector, _lineBasket);
    checkIfParallel(_line, _lineBasket);

    //projection of centerFin point on axis line of basket
    _centerFinProjected = projection(_centerFinPoint, _lineBasket);

    calculateNewPoints(_lineBasket, _centerFinProjected, _centerFinPoint, _centerFin2Point, _centerFin3Point);

    std::vector<pcl::PointXYZ> P{_centerFinPoint, _centerFin2Point, _centerFin3Point};
    std::vector<pcl::PointXYZ> R = movePointsToFinsCenter(P, _centerFinProjected, _finHeight);
    _finsCenter = R;

    _cylinder = buildModelCoefficientCylinder(_centerFinPoint, _centerFinProjected, _basketAxisVector);

    computeTransformation();
}

// estimate the planes for the two surfaces of the fin model.
// The intersection of the two planes will give us the line of the top part of the fin.
// planes is a vector of 2 elements
void DrumModel::findPlanes(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_plane, std::vector<pcl::ModelCoefficients> &planes)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*cloud_plane, *cloud_copy);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointNormal> seg;
    pcl::ModelCoefficients plane;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    for (int i = 0; i < 2; i++)
    {
        seg.setInputCloud(cloud_copy);
        seg.segment(*inliers, plane);

        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        }
        pcl::console::print_highlight("inliers size: %d \n", inliers->indices.size());

        pcl::ExtractIndices<pcl::PointNormal> extract;
        // Extract the inliers
        extract.setInputCloud(cloud_copy);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_copy);

        planes.push_back(plane);
    }
}

// estimate the intersaction line resulting from the two planes.
void DrumModel::estimateIntersactionLine(std::vector<pcl::ModelCoefficients> &planes, pcl::ModelCoefficients &line_model)
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

    std::vector<float> values(&line[0], line.data() + line.cols() * line.rows());
    line_model.values = values;
}

void DrumModel::getPointsOnLine(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud, pcl::ModelCoefficients &line,
                                pcl::PointXYZ &maxFinPoint, pcl::PointXYZ &minFinPoint, pcl::PointXYZ &centerFinPoint)
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

    _finLength = pcl::euclideanDistance<pcl::PointXYZ, pcl::PointXYZ>(maxPoint, minPoint);
    pcl::console::print_highlight("Fin estimated length: %f meters\n", _finLength);

    pcl::PointXYZ midPoint;
    midPoint.x = (maxPoint.x + minPoint.x) / 2;
    midPoint.y = (maxPoint.y + minPoint.y) / 2;
    midPoint.z = (maxPoint.z + minPoint.z) / 2;

    //modify center point
    line.values[0] = midPoint.x;
    line.values[1] = midPoint.y;
    line.values[2] = midPoint.z;

    maxFinPoint = maxPoint;
    minFinPoint = minPoint;
    centerFinPoint = midPoint;
}

//to determine if two lines are parallel in 3D
void DrumModel::checkIfParallel(pcl::ModelCoefficients &line, pcl::ModelCoefficients &axisBasket)
{
    // normalizing the vectors to unit length and computing the norm of the cross-product,
    // which is the sine of the angle between them.

    Eigen::Vector3f vec1;
    vec1.x() = line.values[3];
    vec1.y() = line.values[4];
    vec1.z() = line.values[5];
    vec1.normalize();

    Eigen::Vector3f vec2;
    vec2.x() = axisBasket.values[3];
    vec2.y() = axisBasket.values[4];
    vec2.z() = axisBasket.values[5];
    vec2.normalize();

    Eigen::Vector3f vecCross = vec1.cross(vec2);
    float norm = vecCross.norm();
    if (abs(asin(norm)) > 0.1)
        PCL_WARN("Axes are not parallel");
}

void DrumModel::buildLineModelCoefficient(Eigen::Vector3f &point, Eigen::Vector3f &axis,
                                          pcl::ModelCoefficients &_lineBasket)
{
    std::vector<float> values{point.x(), point.y(), point.z(), axis.x(), axis.y(), axis.z()};
    _lineBasket.values = values;
}

pcl::ModelCoefficients DrumModel::buildModelCoefficientCylinder(pcl::PointXYZ pointFIn, pcl::PointXYZ pointFinProj, Eigen::Vector3f axis)
{
    float distance = pcl::euclideanDistance(pointFIn, pointFinProj);

    std::vector<float> values{pointFinProj.x, pointFinProj.y, pointFinProj.z, axis.x(), axis.y(), axis.z(), distance};
    pcl::ModelCoefficients model;
    model.values = values;
    return model;
}

pcl::PointXYZ DrumModel::projection(pcl::PointXYZ &point, pcl::ModelCoefficients &line)
{
    pcl::PointXYZ linePoint;
    linePoint.x = line.values[0];
    linePoint.y = line.values[1];
    linePoint.z = line.values[2];

    //project points onto the line
    Eigen::Vector3f V;
    V = point.getVector3fMap() - linePoint.getVector3fMap();

    Eigen::Vector3f L{line.values[3], line.values[4], line.values[5]};

    Eigen::Vector3f projectedV;
    projectedV = linePoint.getVector3fMap() + V.dot(L) / L.dot(L) * L;

    pcl::PointXYZ projectedPoint;
    projectedPoint.getVector3fMap() = projectedV;

    return projectedPoint;
}

void DrumModel::calculateNewPoints(pcl::ModelCoefficients &cylinder, pcl::PointXYZ &centerCylinder,
                                   pcl::PointXYZ &centerFin1, pcl::PointXYZ &centerFin2, pcl::PointXYZ &centerFin3)
{
    std::vector<pcl::PointXYZ> newPoints;
    pcl::PointXYZ new_point;
    // Rodrigues' rotation formula

    // angle to rotate
    float theta = (2 * M_PI) / 3.0f;

    // unit versor k
    Eigen::Vector3f k{cylinder.values[3], cylinder.values[4], cylinder.values[5]};
    k.normalize();

    // vector to rotate V
    Eigen::Vector3f V{centerFin1.getVector3fMap() - centerCylinder.getVector3fMap()};
    float normV = V.norm();
    Eigen::Vector3f V_rot;

    // computation of two points, each displaced by 2pi/3
    for (int c = 0; c < 2; c++)
    {

        V_rot = V * cos(theta) + (k.cross(V)) * sin(theta) + k * (k.dot(V)) * (1 - cos(theta));
        V_rot.normalize();

        new_point.x = centerCylinder.x + normV * V_rot.x();
        new_point.y = centerCylinder.y + normV * V_rot.y();
        new_point.z = centerCylinder.z + normV * V_rot.z();

        newPoints.push_back(new_point);
        V = V_rot;
    }

    centerFin2 = newPoints[0];
    centerFin3 = newPoints[1];
}

std::vector<pcl::PointXYZ> DrumModel::movePointsToFinsCenter(std::vector<pcl::PointXYZ> &points, pcl::PointXYZ &center, float distance)
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

float DrumModel::calculateFinHeight(pcl::PointCloud<pcl::PointNormal>::Ptr &input, pcl::ModelCoefficients &line)
{
    std::vector<float> distances;
    float tmp;

    Eigen::Vector4f line_pt(line.values[0], line.values[1], line.values[2], 0);
    Eigen::Vector4f line_dir(line.values[3], line.values[4], line.values[5], 0);
    double sqr = line_dir.norm() * line_dir.norm();

    for (int i = 0; i < input->size(); i++)
    {
        tmp = pcl::sqrPointToLineDistance(input->points[i].getVector4fMap(), line_pt, line_dir, sqr);
        distances.push_back(tmp);
    }

    double max = *std::max_element(distances.begin(), distances.end());
    _finHeight = sqrt(max) / 2;

    pcl::console::print_highlight("Fin estimated height: %f meters\n", _finHeight * 2);
}

void DrumModel::computeTransformation()
{
    Eigen::Vector3f axis2 = _centerFinPoint.getVector3fMap() - _centerFinProjected.getVector3fMap();
    axis2.normalize();

    Eigen::Vector3f axis1 = _basketAxisVector;
    axis1.normalize();

    Eigen::VectorXd from_line_x, from_line_z, to_line_x, to_line_z;

    from_line_x.resize(6);
    from_line_z.resize(6);
    to_line_x.resize(6);
    to_line_z.resize(6);

    //Origin
    from_line_x << 0, 0, 0, 1, 0, 0;
    from_line_z << 0, 0, 0, 0, 0, 1;

    to_line_x.head<3>() = _centerFinProjected.getVector3fMap().cast<double>();
    to_line_x.tail<3>() = axis2.cast<double>();

    to_line_z.head<3>() = _centerFinProjected.getVector3fMap().cast<double>();
    to_line_z.tail<3>() = axis1.cast<double>();

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
    _tSmall0.translation() = _finsCenter[0].getVector3fMap().cast<double>();

    // small cylinder 1
    _tSmall1.linear() = transformation.linear();
    _tSmall1.translation() = _finsCenter[1].getVector3fMap().cast<double>();

    // small cylinder 2
    _tSmall2.linear() = transformation.linear();
    _tSmall2.translation() = _finsCenter[2].getVector3fMap().cast<double>();
}
