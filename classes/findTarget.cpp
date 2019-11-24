#include "../include/findTarget.h"

bool FindTarget::compute()
{
    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    const float leaf = 0.005f;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(object);
    grid.filter(*object);
    grid.setInputCloud(scene);
    grid.filter(*scene);

    // Estimate normals for scene
    pcl::console::print_highlight("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest;
    nest.setNumberOfThreads(4);
    nest.setRadiusSearch(0.01);
    nest.setInputCloud(scene);
    nest.compute(*scene);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch(0.025);
    fest.setInputCloud(object);
    fest.setInputNormals(object);
    fest.compute(*object_features);
    fest.setInputCloud(scene);
    fest.setInputNormals(scene);
    fest.compute(*scene_features);

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
    align.setInputSource(object);
    align.setSourceFeatures(object_features);
    align.setInputTarget(scene);
    align.setTargetFeatures(scene_features);
    align.setMaximumIterations(50000);               // Number of RANSAC iterations
    align.setNumberOfSamples(3);                     // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(5);            // Number of nearest features to use
    align.setSimilarityThreshold(0.95f);             // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(2.5f * leaf); // Inlier threshold
    align.setInlierFraction(0.25f);                  // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align(*object_aligned);
    }

    if (align.hasConverged())
    {
        // Print results
        printf("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation();
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1), transformation(0, 2));
        pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1), transformation(1, 2));
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1), transformation(2, 2));
        pcl::console::print_info("\n");
        pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3), transformation(2, 3));
        pcl::console::print_info("\n");
        pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), object->size());
    }
    else
    {
        pcl::console::print_error("Alignment failed!\n");
        return false;
    }

    pcl::console::print_highlight("Starting ICP alignment ...\n");
    pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setInputSource(object_aligned);
    icp.setInputTarget(scene);
    icp.setMaximumIterations(300);
    icp.setMaxCorrespondenceDistance(0.01);
    icp.align(*object_icp);

    if (icp.hasConverged())
    {
        std::cout << "ICP has converged! score: " << icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;
    }
    else
    {
        return false;
    }

    return true;
}

void FindTarget::visualize(bool spinFlag)
{
    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.setBackgroundColor(1.0, 1.0, 1.0);
    visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 178.0, 0.0), "scene");
    visu.addPointCloud(object_aligned, ColorHandlerT(object_aligned, 0.0, 0.0, 255.0), "object_aligned");

    pcl::visualization::PCLVisualizer visICP("ICP");
    visICP.setBackgroundColor(1.0, 1.0, 1.0);
    visICP.addPointCloud(scene, ColorHandlerT(scene, 0.0, 178.0, 0.0), "scene");
    visICP.addPointCloud(object_icp, ColorHandlerT(object_icp, 0.0, 0.0, 255.0), "object_icp");

    pcl::visualization::PCLVisualizer visFinal("Comparison ICP-Alignment");
    visFinal.setBackgroundColor(1.0, 1.0, 1.0);
    visFinal.addPointCloud(object_icp, ColorHandlerT(object_icp, 0.0, 0.0, 255.0), "final");
    visFinal.addPointCloud(object_aligned, ColorHandlerT(object_aligned, 255.0, 0.0, 0.0), "object_aligned");

    if (spinFlag)
        visu.spin();
}
