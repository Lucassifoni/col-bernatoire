#!/usr/bin/env python3

import os
import sys
import time
import json
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

def load_laz_file(file_path):
    """Load a LAZ/LAS file and return as Open3D point cloud"""
    # Check if file is COPC
    is_copc = '.copc.' in file_path.lower()

    # Method 1: Try PDAL (especially good for COPC)
    try:
        import pdal
        print(f"Loading point cloud using PDAL: {file_path}")

        # Create PDAL pipeline
        pipeline_json = [
            {
                "type": "readers.las",
                "filename": file_path
            }
        ]

        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        # Get data from pipeline
        arrays = pipeline.arrays
        if len(arrays) > 0:
            arr = arrays[0]
            points = np.vstack([arr['X'], arr['Y'], arr['Z']]).transpose()

            # Extract colors if available
            colors = None
            if all(x in arr.dtype.names for x in ['Red', 'Green', 'Blue']):
                colors = np.vstack([
                    arr['Red'] / 65535.0,
                    arr['Green'] / 65535.0,
                    arr['Blue'] / 65535.0
                ]).transpose()

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            print(f"Loaded {len(points)} points using PDAL")
            return pcd

    except Exception as e:
        print(f"Error with PDAL: {e}")

    # Method 2: Try using specialized COPC handling with PDAL
    if is_copc:
        try:
            import pdal
            print("Trying specialized PDAL COPC reader...")

            # Create PDAL pipeline specifically for COPC
            pipeline_json = [
                {
                    "type": "readers.las",
                    "filename": file_path,
                    "use_eb_vlr": True,  # Enable extended binary VLR support for COPC
                    "compression": "lazperf"  # Try using LAZperf decompression
                }
            ]

            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            pipeline.execute()

            # Get data from pipeline
            arrays = pipeline.arrays
            if len(arrays) > 0:
                arr = arrays[0]
                points = np.vstack([arr['X'], arr['Y'], arr['Z']]).transpose()

                # Extract colors if available
                colors = None
                if all(x in arr.dtype.names for x in ['Red', 'Green', 'Blue']):
                    colors = np.vstack([
                        arr['Red'] / 65535.0,
                        arr['Green'] / 65535.0,
                        arr['Blue'] / 65535.0
                    ]).transpose()

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                print(f"Loaded {len(points)} points using specialized PDAL COPC reader")
                return pcd

        except Exception as e:
            print(f"Error with specialized PDAL COPC reader: {e}")

    # Method 3: Try laszip-enabled laspy
    try:
        import laspy
        from laspy import LazBackend

        print("Trying laspy with lazrs backend...")

        # Use lazrs backend for LAZ files
        if file_path.lower().endswith('.laz') or is_copc:
            try:
                with laspy.open(file_path, laz_backend=LazBackend.Lazrs) as fh:
                    las = fh.read()
                    points = np.vstack([las.x, las.y, las.z]).transpose()

                    # Extract colors if available
                    colors = None
                    if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
                        max_red = np.max(las.red) if np.max(las.red) > 0 else 1
                        max_green = np.max(las.green) if np.max(las.green) > 0 else 1
                        max_blue = np.max(las.blue) if np.max(las.blue) > 0 else 1

                        colors = np.vstack([
                            las.red / max_red,
                            las.green / max_green,
                            las.blue / max_blue
                        ]).transpose()

                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    if colors is not None:
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                    print(f"Loaded {len(points)} points using laspy with lazrs backend")
                    return pcd
            except Exception as e:
                print(f"Error with lazrs backend: {e}")

            # Try laszip backend as fallback
            print("Trying with laszip backend...")
            try:
                with laspy.open(file_path, laz_backend=LazBackend.Laszip) as fh:
                    las = fh.read()
                    points = np.vstack([las.x, las.y, las.z]).transpose()

                    # Create Open3D point cloud and add colors if available
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)

                    if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
                        max_red = np.max(las.red) if np.max(las.red) > 0 else 1
                        max_green = np.max(las.green) if np.max(las.green) > 0 else 1
                        max_blue = np.max(las.blue) if np.max(las.blue) > 0 else 1

                        colors = np.vstack([
                            las.red / max_red,
                            las.green / max_green,
                            las.blue / max_blue
                        ]).transpose()

                        pcd.colors = o3d.utility.Vector3dVector(colors)

                    print(f"Loaded {len(points)} points using laspy with laszip backend")
                    return pcd
            except Exception as e:
                print(f"Error with laszip backend: {e}")

        # Standard LAS reading
        else:
            with laspy.open(file_path) as fh:
                las = fh.read()
                points = np.vstack([las.x, las.y, las.z]).transpose()

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
                    max_red = np.max(las.red) if np.max(las.red) > 0 else 1
                    max_green = np.max(las.green) if np.max(las.green) > 0 else 1
                    max_blue = np.max(las.blue) if np.max(las.blue) > 0 else 1

                    colors = np.vstack([
                        las.red / max_red,
                        las.green / max_green,
                        las.blue / max_blue
                    ]).transpose()

                    pcd.colors = o3d.utility.Vector3dVector(colors)

                print(f"Loaded {len(points)} points using laspy")
                return pcd

    except Exception as nested_e:
        print(f"Error with laspy: {nested_e}")

    # Method 4: Try pylas as last resort
    try:
        import pylas
        print("Trying pylas...")

        las = pylas.read(file_path)
        points = np.vstack([las.x, las.y, las.z]).transpose()

        # Extract colors if available
        colors = None
        if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
            max_red = np.max(las.red) if np.max(las.red) > 0 else 1
            max_green = np.max(las.green) if np.max(las.green) > 0 else 1
            max_blue = np.max(las.blue) if np.max(las.blue) > 0 else 1

            colors = np.vstack([
                las.red / max_red,
                las.green / max_green,
                las.blue / max_blue
            ]).transpose()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        print(f"Loaded {len(points)} points using pylas")
        return pcd

    except Exception as e:
        print(f"Error with pylas: {e}")

    # If all methods fail, raise exception
    raise Exception(f"Failed to load point cloud file: {file_path}")

def preprocess_point_cloud(pcd, voxel_size=None, remove_outliers=True):
    """Preprocess the point cloud for better mesh generation"""
    print("Preprocessing point cloud...")

    # Make a copy to avoid modifying the original
    # Use copy method or create a new point cloud with the same points/colors
    try:
        # First try using the copy method (newer Open3D versions)
        processed_pcd = pcd.copy()
    except AttributeError:
        # Fallback for older versions: manually copy the data
        processed_pcd = o3d.geometry.PointCloud()
        processed_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            processed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        if pcd.has_normals():
            processed_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

    # Downsample first if requested (do this before normal estimation)
    if voxel_size is not None:
        print(f"Downsampling with voxel size {voxel_size}...")
        processed_pcd = processed_pcd.voxel_down_sample(voxel_size)

    # Remove outliers if requested (do this before normal estimation)
    if remove_outliers:
        print("Removing outliers...")
        try:
            processed_pcd, _ = processed_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
        except Exception as e:
            print(f"Warning: Outlier removal failed, continuing without it: {e}")

    # Estimate normals if they don't exist
    if not processed_pcd.has_normals():
        print("Estimating normals using robust method...")
        try:
            # Try a more robust normal estimation with smaller search radius
            processed_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
            # Simple orientation that doesn't rely on Qhull
            processed_pcd.orient_normals_towards_camera_location(
                camera_location=np.mean(np.asarray(processed_pcd.points), axis=0) + [0, 0, 1000])
        except Exception as e:
            print(f"Normal estimation failed with error: {e}")
            print("Trying alternative normal estimation method...")
            try:
                # Try even more conservative parameters
                processed_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=9))
                processed_pcd.orient_normals_towards_camera_location(
                    camera_location=np.mean(np.asarray(processed_pcd.points), axis=0) + [0, 0, 1000])
            except Exception as e2:
                print(f"Alternative normal estimation also failed: {e2}")
                print("WARNING: Continuing without proper normals. Mesh quality may be affected.")

    return processed_pcd

def create_mesh(pcd, depth=9, scale=1.1, linear_fit=False):
    """Create a mesh from a point cloud using Poisson surface reconstruction"""
    print(f"Creating mesh (Poisson depth={depth})...")

    # Ensure we have normals
    if not pcd.has_normals():
        print("Warning: Point cloud does not have normals. Attempting to estimate...")
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
            pcd.orient_normals_towards_camera_location(
                camera_location=np.mean(np.asarray(pcd.points), axis=0) + [0, 0, 1000])
        except Exception as e:
            print(f"Failed to estimate normals: {e}")
            print("Attempting mesh creation with ball pivoting instead of Poisson...")
            try:
                radii = [0.05, 0.1, 0.2, 0.4]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))

                # Clean the mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()

                return mesh
            except Exception as e2:
                print(f"Ball pivoting also failed: {e2}")
                raise Exception("Cannot create mesh without valid normals")

    # Create mesh using Poisson surface reconstruction
    with tqdm(total=100, desc="Reconstruction") as pbar:
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=linear_fit)
            pbar.update(100)
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
            print("Trying with reduced depth...")
            try:
                # Try with reduced depth
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=max(5, depth-2), scale=scale, linear_fit=True)
                pbar.update(100)
            except Exception as e2:
                print(f"Reduced depth Poisson also failed: {e2}")
                print("Attempting ball pivoting as a last resort...")
                try:
                    radii = [0.05, 0.1, 0.2, 0.4]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii))
                    pbar.update(100)
                    return mesh
                except Exception as e3:
                    print(f"All mesh creation methods failed: {e3}")
                    raise Exception("Unable to create mesh with available methods")

    # Remove low-density vertices (optional)
    try:
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    except Exception as e:
        print(f"Warning: Could not remove low-density vertices: {e}")

    print(f"Mesh created with {len(mesh.triangles)} triangles")
    return mesh

def crop_to_bounds(mesh, pcd):
    """Crop the mesh to the bounding box of the original point cloud"""
    print("Cropping mesh to original point cloud bounds...")
    bbox = pcd.get_axis_aligned_bounding_box()
    cropped_mesh = mesh.crop(bbox)
    return cropped_mesh

def save_mesh(mesh, output_file):
    """Save the mesh to an STL file"""
    print(f"Saving mesh to {output_file}...")

    # Make sure normals are computed
    if not mesh.has_triangle_normals():
        print("Computing mesh normals...")
        mesh.compute_triangle_normals()

    # Also compute vertex normals for better visualization
    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()

    # Try to ensure the mesh is watertight
    print("Checking mesh integrity...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Save the file
    success = o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=False)

    # Check if saving was successful
    if success:
        print(f"Mesh saved successfully with {len(mesh.triangles)} triangles")
    else:
        raise Exception(f"Failed to save mesh to {output_file}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert LAZ/LAS files to STL for 3D printing')
    parser.add_argument('input_file', help='Input LAZ/LAS file')
    parser.add_argument('--output', '-o', help='Output STL file (default: same as input with .stl extension)')
    parser.add_argument('--depth', '-d', type=int, default=9, help='Poisson reconstruction depth (default: 9)')
    parser.add_argument('--voxel-size', '-v', type=float, help='Voxel size for downsampling (default: None)')
    parser.add_argument('--scale', '-s', type=float, default=1.1, help='Scale factor for Poisson reconstruction (default: 1.1)')
    parser.add_argument('--no-crop', action='store_true', help='Do not crop the mesh to the original point cloud bounds')
    parser.add_argument('--no-outlier-removal', action='store_true', help='Do not remove outliers')
    parser.add_argument('--linear-fit', action='store_true', help='Use linear fit for Poisson reconstruction')
    parser.add_argument('--max-points', type=int, default=None, help='Maximum number of points to process (default: all)')
    parser.add_argument('--reduce-memory', action='store_true', help='Use memory-efficient mode for large point clouds')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        # Handle .copc.laz files correctly
        if base_name.endswith('.copc'):
            base_name = base_name[:-5]  # Remove .copc from the filename
        output_file = f"{base_name}.stl"

    # Start timer
    start_time = time.time()

    try:
        # Load LAZ file
        pcd = load_laz_file(args.input_file)

        # Limit points if requested
        if args.max_points is not None and len(pcd.points) > args.max_points:
            print(f"Limiting to {args.max_points} points (from {len(pcd.points)} total)")

            # Create a new point cloud with random subset of points
            indices = np.random.choice(len(pcd.points), args.max_points, replace=False)
            subset_pcd = o3d.geometry.PointCloud()
            subset_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])

            if pcd.has_colors():
                subset_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])

            pcd = subset_pcd

        # Determine voxel size automatically if reduce-memory is enabled and no voxel size is specified
        if args.reduce_memory and args.voxel_size is None:
            # Estimate a reasonable voxel size based on point cloud density
            points = np.asarray(pcd.points)
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            bbox_size = bbox_max - bbox_min

            # Target around 1-2 million points for memory efficiency
            target_points = 1500000
            current_points = len(pcd.points)

            if current_points > target_points:
                # Calculate voxel size to achieve target point count
                # This is an approximation - we take the cube root of the ratio
                ratio = (current_points / target_points) ** (1/3)

                # Get the smallest dimension for voxel size calculation
                min_dim = min(bbox_size)

                # Set voxel size as a fraction of the smallest dimension
                voxel_size = (min_dim / 1000) * ratio

                print(f"Memory-efficient mode: auto voxel size set to {voxel_size:.4f}")
                args.voxel_size = voxel_size

        # Preprocess the point cloud
        processed_pcd = preprocess_point_cloud(
            pcd,
            voxel_size=args.voxel_size,
            remove_outliers=not args.no_outlier_removal
        )

        # Create mesh
        mesh = create_mesh(
            processed_pcd,
            depth=args.depth,
            scale=args.scale,
            linear_fit=args.linear_fit
        )

        # Crop to bounds if needed
        if not args.no_crop:
            mesh = crop_to_bounds(mesh, pcd)

        # Save the mesh
        save_mesh(mesh, output_file)

        # Print time taken
        elapsed_time = time.time() - start_time
        print(f"Conversion completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
