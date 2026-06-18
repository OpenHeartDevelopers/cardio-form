import os
import random
from scipy.ndimage.morphology import binary_erosion
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from cardio_form.utils import configure_logging

logger = configure_logging(__name__)

# constants
VOXEL_SAMPLING_DIST = 10  # in mm
GRID_SIZE = 160  # in voxels
BB_MARGIN = 1.2  # 20% margin around


# helpers
def norm_vector(v):
    nrm_v = np.linalg.norm(v)
    return v / nrm_v, nrm_v


def extract_slice_label_coordinates(segmentation_sax):
    """
    Extract coordinates for each label in short-axis (SAX) segmentation slices.

    Args:
        segmentation_sax: 3D array where last dimension represents different slice planes

    Returns:
        tuple: (slice_coordinates, slice_labels)
            - slice_coordinates: List of lists, where each inner list contains coordinate
              arrays for each label in that slice
            - slice_labels: List of arrays containing unique label values per slice
    """
    num_slices = np.shape(segmentation_sax)[-1]
    slice_coordinates = []
    slice_labels = []

    # Process each slice plane independently
    for slice_idx in range(num_slices):
        # Extract single 2D slice
        current_slice = segmentation_sax[..., slice_idx]

        # Find all unique label values in this slice
        unique_labels = np.unique(current_slice)
        slice_labels.append(unique_labels)

        # Extract coordinates for each label in this slice
        label_coords_in_slice = []
        for label_value in unique_labels:
            # Create binary mask for current label
            label_mask = current_slice == label_value

            # Get (x, y) coordinates where this label exists
            coords = np.array(np.where(label_mask[..., np.newaxis]))

            # Add slice index as z-coordinate
            coords[-1] = slice_idx

            label_coords_in_slice.append(coords)

        slice_coordinates.append(label_coords_in_slice)

    return slice_coordinates, slice_labels


def extract_longaxis_label_coordinates(segmentation_lax):
    """
    Extract coordinates for each label in long-axis (LAX) segmentation.

    Args:
        segmentation_lax: 3D array with long-axis segmentation (uses first slice only)

    Returns:
        list: Coordinate arrays for each unique label found in the first LAX slice
    """
    # Extract first long-axis slice
    lax_slice = segmentation_lax[..., 0]

    # Find all unique label values
    unique_labels = np.unique(lax_slice)

    # Extract coordinates for each label
    label_coordinates = []
    for label_value in unique_labels:
        # Create binary mask for current label
        label_mask = lax_slice == label_value

        # Get (x, y) coordinates where this label exists
        coords = np.where(label_mask[..., np.newaxis])

        label_coordinates.append(coords)

    return label_coordinates


def compute_lax_plane_geometry(lax_segmentation, lax_affine_tx):
    lax_segmentation_coords = extract_longaxis_label_coordinates(lax_segmentation)

    lax_point_coordinates = []
    for coordinate in lax_segmentation_coords:
        ijk = np.array(coordinate)
        ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
        xyz1 = np.matmul(lax_affine_tx, ijk1)
        xyz = xyz1[0:3, ...]
        lax_point_coordinates.append(xyz)

    # ipp, ipo
    im_plane_pos = np.matmul(lax_affine_tx, np.array([[0], [0], [0], [1]]))[0:3]
    im_plane_0 = np.matmul(
        lax_affine_tx, np.array([[VOXEL_SAMPLING_DIST], [0], [0], [1]])
    )[0:3]
    im_plane_1 = np.matmul(
        lax_affine_tx, np.array([[0], [VOXEL_SAMPLING_DIST], [0], [1]])
    )[0:3]

    v0, norm_v0 = norm_vector(im_plane_0 - im_plane_pos)
    v1, norm_v1 = norm_vector(im_plane_1 - im_plane_pos)

    im_plane_orientation = np.concatenate((v0, v1))

    pixel_spacing = [norm_v0 / VOXEL_SAMPLING_DIST, norm_v1 / VOXEL_SAMPLING_DIST]

    return (
        lax_point_coordinates,
        im_plane_pos,
        im_plane_orientation,
        pixel_spacing,
        lax_segmentation,
    )


def compute_sax_plane_geometry(sax_segmentation, sax_affine_tx):
    sax_slice_coords, sax_slice_labels = extract_slice_label_coordinates(
        sax_segmentation
    )

    sax_slice_point_coords = []
    for label_coords_in_slice in sax_slice_coords:
        points_in_slice = []
        for coord in label_coords_in_slice:
            ijk = np.array(coord)
            ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
            xyz1 = np.matmul(sax_affine_tx, ijk1)
            xyz = xyz1[0:3, ...]
            points_in_slice.append(xyz)
        sax_slice_point_coords.append(points_in_slice)

    # Calculate plane position for each slice
    slice_plane_positions = []
    num_slices = np.shape(sax_segmentation)[-1]
    for ks in range(num_slices):
        position = np.matmul(sax_affine_tx, np.array([[0], [0], [ks], [1]]))[0:3]
        slice_plane_positions.append(position)

    im_plane_0 = np.matmul(
        sax_affine_tx, np.array([[VOXEL_SAMPLING_DIST], [0], [0], [1]])
    )[0:3]
    im_plane_1 = np.matmul(
        sax_affine_tx, np.array([[0], [VOXEL_SAMPLING_DIST], [0], [1]])
    )[0:3]

    v0, norm_v0 = norm_vector(im_plane_0 - slice_plane_positions[0])
    v1, norm_v1 = norm_vector(im_plane_1 - slice_plane_positions[0])

    plane_orientation = np.concatenate((v0, v1))

    pixel_spacing = [norm_v0 / VOXEL_SAMPLING_DIST, norm_v1 / VOXEL_SAMPLING_DIST]

    return (
        sax_slice_point_coords,
        sax_slice_labels,
        slice_plane_positions,
        plane_orientation,
        pixel_spacing,
        sax_segmentation,
    )


# data processing - CONTOURS
def extract_slice_label_contours(segmentation_sax):
    """
    Extract contour coordinates for each non-zero label in short-axis slices.

    Args:
        segmentation_sax: 3D array where last dimension represents different slice planes

    Returns:
        tuple: (slice_contours, slice_labels)
            - slice_contours: List of lists containing contour coordinates for each label
            - slice_labels: List of arrays containing non-zero label values per slice
    """
    num_slices = np.shape(segmentation_sax)[-1]
    slice_contours = []
    slice_labels = []

    for slice_idx in range(num_slices):
        current_slice = segmentation_sax[..., slice_idx]

        # Find all unique labels, excluding background (0)
        unique_labels = np.unique(current_slice)
        non_zero_labels = [label for label in unique_labels if label > 0]
        slice_labels.append(non_zero_labels)

        contours_in_slice = []
        for label_value in non_zero_labels:
            # Create binary mask for current label
            label_mask = current_slice == label_value

            # Extract contour by subtracting eroded mask from original
            # This gives boundary pixels only
            eroded_mask = binary_erosion(label_mask).astype(int)
            contour_mask = np.subtract(label_mask, eroded_mask)

            # Get coordinates of contour pixels
            coords = np.array(np.where(contour_mask[..., np.newaxis]))
            coords[-1] = slice_idx  # Add slice index as z-coordinate

            contours_in_slice.append(coords)

        slice_contours.append(contours_in_slice)

    return slice_contours, slice_labels


def extract_longaxis_label_contours(segmentation_lax):
    """
    Extract contour coordinates for each non-zero label in first long-axis slice.

    Args:
        segmentation_lax: 3D array with long-axis segmentation

    Returns:
        list: Contour coordinate arrays for each non-zero label
    """
    lax_slice = segmentation_lax[..., 0]

    # Find all unique labels, excluding background (0)
    unique_labels = np.unique(lax_slice)
    non_zero_labels = [label for label in unique_labels if label > 0]

    label_contours = []
    for label_value in non_zero_labels:
        # Create binary mask for current label
        label_mask = lax_slice == label_value

        # Extract contour by subtracting eroded mask from original
        # This gives boundary pixels only
        eroded_mask = binary_erosion(label_mask).astype(int)
        contour_mask = np.subtract(label_mask, eroded_mask)

        # Get coordinates of contour pixels
        coords = np.where(contour_mask[..., np.newaxis])

        label_contours.append(coords)

    return label_contours


def compute_lax_contour_geometry(lax_segmentation, lax_affine_tx):
    lax_contour_coords = extract_longaxis_label_contours(lax_segmentation)
    lax_contour_point_coords = []
    for coord in lax_contour_coords:
        ijk = np.array(coord)
        ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
        xyz1 = np.matmul(lax_affine_tx, ijk1)
        xyz = xyz1[0:3, ...]
        lax_contour_point_coords.append(xyz)

    return lax_contour_point_coords, lax_affine_tx


def compute_sax_contour_geometry(sax_segmentation, sax_affine_tx):
    sax_slice_contours, sax_slice_labels = extract_slice_label_contours(
        sax_segmentation
    )
    sax_slice_contour_points = []
    for slice_contours in sax_slice_contours:
        sax_pc = []
        for contour_coords in slice_contours:
            ijk = np.array(contour_coords)
            ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
            xyz1 = np.matmul(sax_affine_tx, ijk1)
            xyz = xyz1[0:3, ...]
            sax_pc.append(xyz)
        sax_slice_contour_points.append(sax_pc)
    return sax_slice_contour_points, sax_slice_labels, sax_affine_tx


def rand_tri(ch2_ps):
    """
    Compute normal vector from a triangle formed by three points.

    By default, uses deterministic behavior (starts from first point).
    To enable original random behavior, set environment variable:

    Usage (to enable random):
        export USE_RANDOM_TRI=True  # bash/zsh
        set USE_RANDOM_TRI=True     # Windows

    Args:
        ch2_ps: Point cloud data

    Returns:
        Normalized normal vector
    """
    # Read from environment, default to False (deterministic)
    # Only uses random if explicitly set to True
    use_random = os.getenv("USE_RANDOM_TRI", "False").lower() == "true"

    if use_random:
        i0 = random.choice(list(range(len(ch2_ps[0][0]))))
    else:
        i0 = 0

    points = np.transpose(ch2_ps[0])

    x0 = points[i0]
    d0 = np.linalg.norm(points - x0, axis=1)

    i1 = np.argmax(d0)
    x1 = points[i1]
    d1 = np.linalg.norm(points - x1, axis=1)

    i2 = np.argmax(d0 + d1)
    x2 = points[i2]

    n = np.cross(x2 - x0, x1 - x0) / np.linalg.norm(np.cross(x2 - x0, x1 - x0))
    return n


def compute_grid_geometry(
    ch2_plane_geom, ch4_plane_geom, ch2_point_coords, ch4_point_coords
):
    def get_axis_direction(vec1, vec2, plane_geom):
        axis_initial = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))
        vec_at_plane_0 = np.mean(plane_geom[1], axis=1)
        vec_at_plane_1 = np.mean(plane_geom[3], axis=1)
        if np.dot(vec_at_plane_1 - vec_at_plane_0, axis_initial) > 0:
            axis_direction = axis_initial
        else:
            axis_direction = -axis_initial
        return axis_direction

    # ax
    normal_2ch = rand_tri(ch2_plane_geom)
    normal_4ch = rand_tri(ch4_plane_geom)

    axis_apex2base = get_axis_direction(normal_2ch, normal_4ch, ch2_plane_geom)
    axis_left2right = get_axis_direction(normal_4ch, axis_apex2base, ch4_plane_geom)
    axis_front2back = np.cross(axis_apex2base, axis_left2right) / np.linalg.norm(
        np.cross(axis_apex2base, axis_left2right)
    )

    # og
    all_coordinates = []
    for ch2_coord_set in ch2_point_coords:
        for point_idx in range(np.shape(ch2_coord_set)[1]):
            all_coordinates.append(np.array(ch2_coord_set)[:, point_idx])

    for ch4_coord_set in ch4_point_coords:
        for point_idx in range(np.shape(ch4_coord_set)[1]):
            all_coordinates.append(np.array(ch4_coord_set)[:, point_idx])

    min_lax = np.min(all_coordinates, axis=0)
    max_lax = np.max(all_coordinates, axis=0)
    origin_centre = np.mean((min_lax, max_lax), axis=0)
    centered_coords = all_coordinates - origin_centre

    # Compute projections, dimensions, and center offsets for each axis
    axes = [axis_apex2base, axis_left2right, axis_front2back]
    dimensions = []
    center_offsets = []

    for axis in axes:
        # Project all points onto this axis once
        projections = np.dot(centered_coords, axis)

        # Get min and max of projections
        proj_min = projections.min()
        proj_max = projections.max()

        # Calculate dimension and center offset
        dimension = proj_max - proj_min
        center_offset = (proj_max + proj_min) / 2

        dimensions.append(dimension)
        center_offsets.append(center_offset)

    # Unpack for clarity (or keep as lists if preferred)
    d_ab, d_lr, d_fb = dimensions
    c_ab, c_lr, c_fb = center_offsets

    origin_adj = (
        origin_centre
        + c_ab * axis_apex2base
        + c_lr * axis_left2right
        + c_fb * axis_front2back
    )
    vs = np.max([d_ab, d_lr, d_fb]) * BB_MARGIN / GRID_SIZE  # 10% margin around
    # vol dense
    ii = np.linspace(0, int(GRID_SIZE - 1), GRID_SIZE) - int(GRID_SIZE / 2)
    jj = np.linspace(0, int(GRID_SIZE - 1), GRID_SIZE) - int(GRID_SIZE / 2)
    kk = np.linspace(0, int(GRID_SIZE - 1), GRID_SIZE) - int(GRID_SIZE / 2)

    iv, jv, kv = np.meshgrid(ii, jj, kk)
    voxel_indices = np.array(
        [
            np.resize(iv, np.size(iv)),
            np.resize(jv, np.size(jv)),
            np.resize(kv, np.size(kv)),
        ]
    ).transpose()
    rotation_matrix = np.array(
        [axis_apex2base, axis_left2right, axis_front2back]
    ).transpose()
    world_coords = (
        origin_adj + np.dot(rotation_matrix, voxel_indices.transpose()).transpose() * vs
    )

    return world_coords, voxel_indices, rotation_matrix, vs, normal_2ch, normal_4ch


def vol_grid_gen(
    ch2_ps,
    ch4_ps,
    sax_ps,
    ch2_pc,
    ch4_pc,
    sax_pc,
    sax_ipp,
    sax_ipo,
    sax_pxs,
    sax_lab,
    ch2_ipp,
    ch2_ipo,
    ch2_pxs,
    ch2_lab,
    ch4_ipp,
    ch4_ipo,
    ch4_pxs,
    ch4_lab,
):
    xyz_v, ijk_v, vm, vs, n_2ch, n_4ch = compute_grid_geometry(
        ch2_ps, ch4_ps, ch2_pc, ch4_pc
    )
    # vol sparse
    vol_sp = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    ijk_v_ = ijk_v + 80
    affine_3d = np.eye(4)
    affine_3d[0:3, 0:3] = vm * vs
    affine_3d[0:3, 3] = xyz_v[0, :]
    # 2ch
    d2 = np.dot(xyz_v - ch2_pc[0][:, 0], n_2ch)
    i_2ch = np.where(np.abs(d2) <= vs)[0]
    xyz_2ch = xyz_v[i_2ch, :]
    v0_s = ch2_ipo[0:3]
    v1_s = ch2_ipo[3:]
    pq_2ch = (
        np.transpose(
            [
                np.dot(xyz_2ch - np.transpose(ch2_ipp), v0_s) / ch2_pxs[0],
                np.dot(xyz_2ch - np.transpose(ch2_ipp), v1_s) / ch2_pxs[1],
            ]
        )
        .squeeze()
        .round()
    )
    px2vx_2ch = [0, 1, 2, 5]

    out_of_bounds_counter = 0
    total_voxels = len(i_2ch)
    for ki in range(len(i_2ch)):
        try:
            vol_sp[
                ijk_v_[i_2ch[ki]][0].astype(int),
                ijk_v_[i_2ch[ki]][1].astype(int),
                ijk_v_[i_2ch[ki]][2].astype(int),
            ] = px2vx_2ch[
                ch2_lab[pq_2ch[ki][0].astype(int), pq_2ch[ki][1].astype(int), 0].astype(
                    int
                )
            ]
        except:
            out_of_bounds_counter += 1
    if out_of_bounds_counter > 0:
        logger.warning(
            f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
            f"were projected out of the target volume's bounds. This is usually acceptable."
        )
    # 4ch
    d4 = np.dot(xyz_v - ch4_pc[0][:, 0], n_4ch)
    i_4ch = np.where(np.abs(d4) <= vs)[0]
    xyz_4ch = xyz_v[i_4ch, :]
    v0_s = ch4_ipo[0:3]
    v1_s = ch4_ipo[3:]
    pq_4ch = (
        np.transpose(
            [
                np.dot(xyz_4ch - np.transpose(ch4_ipp), v0_s) / ch4_pxs[0],
                np.dot(xyz_4ch - np.transpose(ch4_ipp), v1_s) / ch4_pxs[1],
            ]
        )
        .squeeze()
        .round()
    )
    px2vx_4ch = [0, 1, 2, 3, 5, 6]

    out_of_bounds_counter = 0
    total_voxels = len(i_4ch)
    for ki in range(len(i_4ch)):
        try:
            vol_sp[
                ijk_v_[i_4ch[ki]][0].astype(int),
                ijk_v_[i_4ch[ki]][1].astype(int),
                ijk_v_[i_4ch[ki]][2].astype(int),
            ] = px2vx_4ch[
                ch4_lab[pq_4ch[ki][0].astype(int), pq_4ch[ki][1].astype(int), 0].astype(
                    int
                )
            ]
        except:
            out_of_bounds_counter += 1
    if out_of_bounds_counter > 0:
        logger.warning(
            f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
            f"were projected out of the target volume's bounds. This is usually acceptable."
        )
    # sax
    ns = len(sax_pc)
    for ks in range(ns):
        sax_ps_ks = sax_ps[ks]
        n_ks = rand_tri(sax_ps_ks)
        d_ks = np.dot(xyz_v - sax_ps_ks[0][:, 0], n_ks)
        i_ks = np.where(np.abs(d_ks) <= vs)[0]
        xyz_ks = xyz_v[i_ks, :]
        v0_s = sax_ipo[0:3]
        v1_s = sax_ipo[3:]
        pq_sax = (
            np.transpose(
                [
                    np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v0_s) / sax_pxs[0],
                    np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v1_s) / sax_pxs[1],
                ]
            )
            .squeeze()
            .round()
        )
        px2vx_sax = [0, 1, 2, 3]

        out_of_bounds_counter = 0
        total_voxels = len(i_ks)
        for ki in range(len(i_ks)):
            try:
                if (
                    sax_lab[pq_sax[ki][0].astype(int), pq_sax[ki][1].astype(int), ks]
                    > 0
                ):
                    vol_sp[
                        ijk_v_[i_ks[ki]][0].astype(int),
                        ijk_v_[i_ks[ki]][1].astype(int),
                        ijk_v_[i_ks[ki]][2].astype(int),
                    ] = px2vx_sax[
                        sax_lab[
                            pq_sax[ki][0].astype(int), pq_sax[ki][1].astype(int), ks
                        ].astype(int)
                    ]
            except:
                out_of_bounds_counter += 1
        if out_of_bounds_counter > 0:
            logger.warning(
                f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
                f"were projected out of the target volume's bounds. This is usually acceptable."
            )
    return vol_sp, affine_3d


def vol_grid_bp(
    ch2_ps,
    ch4_ps,
    sax_ps,
    ch2_pc,
    ch4_pc,
    sax_pc,
    sax_ipp,
    sax_ipo,
    sax_pxs,
    sax_lab,
    ch2_ipp,
    ch2_ipo,
    ch2_pxs,
    ch2_lab,
    ch4_ipp,
    ch4_ipo,
    ch4_pxs,
    ch4_lab,
    vol_pr,
):
    xyz_v, ijk_v, vm, vs, n_2ch, n_4ch = compute_grid_geometry(
        ch2_ps, ch4_ps, ch2_pc, ch4_pc
    )
    # vol dense
    vol_ds = vol_pr
    ijk_v_ = ijk_v + 80
    # 2ch
    d2 = np.dot(xyz_v - ch2_pc[0][:, 0], n_2ch)
    i_2ch = np.where(np.abs(d2) <= vs)[0]
    xyz_2ch = xyz_v[i_2ch, :]
    v0_s = ch2_ipo[0:3]
    v1_s = ch2_ipo[3:]
    pq_2ch = (
        np.transpose(
            [
                np.dot(xyz_2ch - np.transpose(ch2_ipp), v0_s) / ch2_pxs[0],
                np.dot(xyz_2ch - np.transpose(ch2_ipp), v1_s) / ch2_pxs[1],
            ]
        )
        .squeeze()
        .round()
    )
    ch2_bp = np.zeros(np.shape(ch2_lab))
    px2vx_2ch = [0, 1, 2, 0, 0, 3, 0, 0, 0]

    out_of_bounds_counter = 0
    total_voxels = len(i_2ch)
    for ki in range(len(i_2ch)):
        try:
            ch2_bp[pq_2ch[ki][0].astype(int), pq_2ch[ki][1].astype(int), 0] = px2vx_2ch[
                vol_ds[
                    ijk_v_[i_2ch[ki]][0].astype(int),
                    ijk_v_[i_2ch[ki]][1].astype(int),
                    ijk_v_[i_2ch[ki]][2].astype(int),
                ].astype(int)
            ]
        except:
            out_of_bounds_counter += 1

    if out_of_bounds_counter > 0:
        logger.warning(
            f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
            f"were projected out of the target volume's bounds. This is usually acceptable."
        )

    # 4ch
    d4 = np.dot(xyz_v - ch4_pc[0][:, 0], n_4ch)
    i_4ch = np.where(np.abs(d4) <= vs)[0]
    xyz_4ch = xyz_v[i_4ch, :]
    v0_s = ch4_ipo[0:3]
    v1_s = ch4_ipo[3:]
    pq_4ch = (
        np.transpose(
            [
                np.dot(xyz_4ch - np.transpose(ch4_ipp), v0_s) / ch4_pxs[0],
                np.dot(xyz_4ch - np.transpose(ch4_ipp), v1_s) / ch4_pxs[1],
            ]
        )
        .squeeze()
        .round()
    )
    ch4_bp = np.zeros(np.shape(ch2_lab))
    px2vx_4ch = [0, 1, 2, 3, 0, 4, 5, 0, 0]

    out_of_bounds_counter = 0
    total_voxels = len(i_4ch)
    for ki in range(len(i_4ch)):
        try:
            ch4_bp[pq_4ch[ki][0].astype(int), pq_4ch[ki][1].astype(int), 0] = px2vx_4ch[
                vol_ds[
                    ijk_v_[i_4ch[ki]][0].astype(int),
                    ijk_v_[i_4ch[ki]][1].astype(int),
                    ijk_v_[i_4ch[ki]][2].astype(int),
                ].astype(int)
            ]
        except:
            out_of_bounds_counter += 1
    if out_of_bounds_counter > 0:
        logger.warning(
            f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
            f"were projected out of the target volume's bounds. This is usually acceptable."
        )

    # sax
    ns = len(sax_pc)
    sax_bp = np.zeros(np.shape(sax_lab))
    for ks in range(ns):
        sax_ps_ks = sax_ps[ks]
        n_ks = rand_tri(sax_ps_ks)
        d_ks = np.dot(xyz_v - sax_ps_ks[0][:, 0], n_ks)
        i_ks = np.where(np.abs(d_ks) <= vs)[0]
        xyz_ks = xyz_v[i_ks, :]
        v0_s = sax_ipo[0:3]
        v1_s = sax_ipo[3:]
        pq_sax = (
            np.transpose(
                [
                    np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v0_s) / sax_pxs[0],
                    np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v1_s) / sax_pxs[1],
                ]
            )
            .squeeze()
            .round()
        )
        px2vx_sax = [0, 1, 2, 3, 0, 0, 0, 0, 0]

        out_of_bounds_counter = 0
        total_voxels = len(i_ks)
        for ki in range(len(i_ks)):
            try:
                sax_bp[pq_sax[ki][0].astype(int), pq_sax[ki][1].astype(int), ks] = (
                    px2vx_sax[
                        vol_ds[
                            ijk_v_[i_ks[ki]][0].astype(int),
                            ijk_v_[i_ks[ki]][1].astype(int),
                            ijk_v_[i_ks[ki]][2].astype(int),
                        ].astype(int)
                    ]
                )
            except:
                out_of_bounds_counter += 1
        if out_of_bounds_counter > 0:
            logger.warning(
                f"Notice: {out_of_bounds_counter} out of {total_voxels} voxels "
                f"were projected out of the target volume's bounds. This is usually acceptable."
            )

    return ch2_bp.astype(np.uint8), ch4_bp.astype(np.uint8), sax_bp.astype(np.uint8)


## post processing helpers
def resample_to_lps(nifti_image: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Resamples a NIfTI image with oblique orientation to standard LPS orientation.

    This function performs a full geometric resampling. It correctly handles the
    origin, spacing, and direction matrix of the input image to ensure that the
    resampled output is geometrically accurate and aligned with the LPS axes.

    Parameters:
    -----------
    nifti_image : nib.Nifti1Image
        Input NIfTI image with potentially oblique orientation.

    Returns:
    --------
    nib.Nifti1Image
        Resampled image in canonical LPS orientation with an axis-aligned affine matrix.
    """
    # Extract original image properties. CRITICAL: Cast to native Python types.
    spacing = tuple(float(z) for z in nifti_image.header.get_zooms()[:3])
    origin = tuple(float(c) for c in nifti_image.affine[:3, 3])
    size = tuple(int(s) for s in nifti_image.shape[:3])

    # --- THE CRITICAL FIX: Extract the Direction Matrix ---
    # The affine is composed of: [R*S | t], where R is rotation, S is scaling, t is translation.
    # We need to extract the pure rotation component (the direction cosines).
    affine_3x3 = nifti_image.affine[:3, :3]
    direction_matrix = np.zeros((3, 3))
    for i in range(3):
        col = affine_3x3[:, i]
        # Normalize each column vector to get the direction cosine
        direction_matrix[:, i] = col / np.linalg.norm(col)

    # SimpleITK expects the direction matrix in a flattened, row-major tuple.
    direction_tuple = tuple(direction_matrix.T.flatten())

    # --- Create SimpleITK Image with FULL Geometric Information ---
    # Note: Nibabel data is (x, y, z), SimpleITK expects array data as (z, y, x)
    sitk_image = sitk.GetImageFromArray(nifti_image.get_fdata().transpose(2, 1, 0))
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction_tuple)  # <-- THIS WAS THE MISSING PIECE

    # --- Define the Target LPS Grid ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Identity = LPS
    resampler.SetOutputOrigin(origin)  # Keep the same physical origin
    resampler.SetOutputSpacing(spacing)  # Keep the same voxel spacing
    resampler.SetSize(size)  # Keep the same image dimensions
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())  # We are resampling in place

    # Execute resampling
    resampled_sitk = resampler.Execute(sitk_image)

    # --- Convert Back to Nibabel ---
    resampled_data = sitk.GetArrayFromImage(resampled_sitk).transpose(2, 1, 0)

    # Create the new, clean, axis-aligned LPS affine matrix
    new_affine = np.eye(4)
    new_affine[0, 0] = spacing[0]
    new_affine[1, 1] = spacing[1]
    new_affine[2, 2] = spacing[2]
    new_affine[:3, 3] = (
        resampled_sitk.GetOrigin()
    )  # Get the origin from the resampled image

    return nib.Nifti1Image(
        resampled_data.astype(nifti_image.get_data_dtype()), new_affine
    )


def filter_labels(data: np.ndarray, labels_to_keep: list) -> np.ndarray:
    """
    Keep only a specific set of labels in a segmentation array.

    Pure transform: returns a new array containing only the voxels whose value
    is in ``labels_to_keep``; everything else becomes background (0). File I/O
    is the caller's responsibility (see ``cardio_form.io``).

    Args:
        data (np.ndarray): Integer segmentation label map.
        labels_to_keep (list): Label values to retain.

    Returns:
        np.ndarray: Filtered label map (same shape/dtype as ``data``).
    """
    logger.info(f"Filtering segmentation to keep labels: {labels_to_keep}")

    filtered_data = np.zeros_like(data)
    for label in labels_to_keep:
        filtered_data[data == label] = label

    return filtered_data


def merge_labels(
    data: np.ndarray, labels_to_merge: list, value_after_merge: int = None
) -> np.ndarray:
    """
    Merge a set of labels into a single value in a segmentation array.

    Pure transform: returns a new array where every voxel whose value is in
    ``labels_to_merge`` is set to ``value_after_merge``. File I/O is the
    caller's responsibility (see ``cardio_form.io``).

    Args:
        data (np.ndarray): Integer segmentation label map.
        labels_to_merge (list): Label values to merge.
        value_after_merge (int, optional): Value to assign to the merged labels.
            Defaults to the first entry in ``labels_to_merge``.

    Returns:
        np.ndarray: Merged label map (same shape/dtype as ``data``).
    """
    logger.info(f"Merging segmentation labels: {labels_to_merge}")

    if value_after_merge is None:
        value_after_merge = labels_to_merge[0]
        logger.info(
            f"No value_after_merge provided, using the first label ({value_after_merge})."
        )

    merged_data = data.copy()
    for label in labels_to_merge:
        merged_data[data == label] = value_after_merge

    return merged_data


def remap_labels(data: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Remap label values according to an explicit ``{old: new}`` mapping.

    Pure transform: returns a new array where each voxel whose value is a key in
    ``mapping`` is replaced by the corresponding value. Masks are computed
    against the original ``data``, so remappings never cascade (a value remapped
    to something that is itself a key is not remapped again). Labels not present
    in ``mapping`` are left unchanged. File I/O is the caller's responsibility
    (see ``cardio_form.io``).

    Args:
        data (np.ndarray): Integer segmentation label map.
        mapping (dict): ``{old_value: new_value}`` integer mapping.

    Returns:
        np.ndarray: Remapped label map (same shape/dtype as ``data``).
    """
    logger.info(f"Remapping segmentation labels: {mapping}")

    remapped_data = data.copy()
    for old_value, new_value in mapping.items():
        remapped_data[data == old_value] = new_value

    return remapped_data
