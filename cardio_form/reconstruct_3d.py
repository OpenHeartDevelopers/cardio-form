import os
import random

import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_erosion

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from cardio_form.output_managers import ReconstructOutputManager
from cardio_form.utils import check_file_exists

# network
class ReconstructUNet3D(nn.Module):
    def contracting_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.Sigmoid()
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(ReconstructUNet3D, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channel, 16, 32)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(128),
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=256, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(128+256, 128, 128)
        self.conv_decode2 = self.expansive_block(64+128, 64, 64)
        self.final_layer = self.final_block(32+64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=False)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)
        final_layer = self.final_layer(decode_block1)
        return final_layer

SEG_MODE_CHOICES = ['contour', 'plane']
def extract_label_coordinates(labeled_image, mode='contour'):
    """
    Extract voxel coordinates for each label in a labeled image.
    
    Parameters:
    -----------
    labeled_image : ndarray
        Labeled image with shape (..., n_slices).
    mode : str
        'contour' - extract only boundary voxels, exclude label 0
        'plane' - extract all voxels, include all labels
    
    Returns:
    --------
    coordinates : list of lists
        coordinates[slice_idx][label_idx] contains voxel coordinates (ndarray).
    labels : list of lists
        labels[slice_idx] contains label values for that slice.
    """
    
    if mode not in SEG_MODE_CHOICES: 
        raise ValueError(f'MODE NOT RECOGNISED: {mode}.\n Choose between "contour" and "plane"')
    
    ns = np.shape(labeled_image)[-1]
    all_coordinates = []
    all_labels = []
    
    for ks in range(ns):
        slice_data = labeled_image[..., ks]
        labs = np.unique(slice_data)
        
        # Filter labels based on mode
        if mode == 'contour':
            labs = [x for x in labs if x > 0]
        else:  # mode == 'plane'
            labs = list(labs)
        
        all_labels.append(labs)
        slice_coords = []
        
        for kl in labs:
            bm_i = slice_data == kl
            
            # Extract boundary or all voxels based on mode
            if mode == 'contour':
                bm_i = np.subtract(bm_i, binary_erosion(bm_i).astype(int))
            
            # Get coordinates
            pc_i = np.array(np.where(bm_i[..., np.newaxis]))
            
            # Set slice index for multi-slice data
            if ns > 1:
                pc_i[-1] = ks
            
            slice_coords.append(pc_i)
        
        all_coordinates.append(slice_coords)
    
    return all_coordinates, all_labels

def transform_to_world_coordinates(voxel_coords, affine):
    """
    Transform voxel coordinates to world coordinates using affine matrix.
    
    Parameters:
    -----------
    voxel_coords : list of lists
        voxel_coords[slice_idx][label_idx] contains voxel coordinates.
    affine : ndarray
        4x4 affine transformation matrix.
    
    Returns:
    --------
    world_coords : list of lists
        world_coords[slice_idx][label_idx] contains xyz coordinates (3, n_points).
    """
    world_coords = []
    
    for slice_coords in voxel_coords:
        slice_world = []
        for coords in slice_coords:
            ijk = np.array(coords)
            ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
            xyz1 = np.matmul(affine, ijk1)
            xyz = xyz1[0:3, ...]
            slice_world.append(xyz)
        world_coords.append(slice_world)
    
    return world_coords

def compute_plane_geometry(affine, labeled_image=None):
    """
    Compute plane geometry: image plane position (ipp), orientation (ipo), and pixel spacing (pxs).
    
    Parameters:
    -----------
    affine : ndarray
        4x4 affine transformation matrix.
    labeled_image : ndarray, optional
        Labeled image with shape (..., n_slices). Required for multi-slice (SAX).
    
    Returns:
    --------
    ipp : ndarray or list of ndarrays
        Image plane position(s). Single (3,1) array for LAX, list of (3,1) arrays for SAX.
    ipo : ndarray
        Image plane orientation (6,1) - concatenated row and column direction vectors.
    pxs : list
        Pixel spacing [row_spacing, col_spacing].
    """
    # Calculate reference points
    ipp_origin = np.matmul(affine, np.array([[0], [0], [0], [1]]))[0:3]
    ip0 = np.matmul(affine, np.array([[10], [0], [0], [1]]))[0:3]
    ip1 = np.matmul(affine, np.array([[0], [10], [0], [1]]))[0:3]
    
    # Calculate orientation vectors
    v0 = (ip0 - ipp_origin) / np.linalg.norm(ip0 - ipp_origin)
    v1 = (ip1 - ipp_origin) / np.linalg.norm(ip1 - ipp_origin)
    ipo = np.concatenate((v0, v1))
    
    # Calculate pixel spacing
    pxs = [np.linalg.norm(ip0 - ipp_origin) / 10, 
           np.linalg.norm(ip1 - ipp_origin) / 10]
    
    # Calculate ipp for each slice if multi-slice
    if labeled_image is not None and np.shape(labeled_image)[-1] > 1:
        ns = np.shape(labeled_image)[-1]
        ipp = [np.matmul(affine, np.array([[0], [0], [ks], [1]]))[0:3] 
               for ks in range(ns)]
    else:
        ipp = ipp_origin
    
    return ipp, ipo, pxs

# ============================================================================
# HIGH-LEVEL API FUNCTIONS - DATA LOADING
# ============================================================================

def load_contours(nifti_file):
    """
    Load contours (boundaries only) from a NIfTI file.
    
    Parameters:
    -----------
    nifti_file : str
        Path to NIfTI file (SAX or LAX).
    
    Returns:
    --------
    contours : list of lists
        contours[slice_idx][label_idx] contains xyz coordinates (3, n_points).
    labels : list of lists
        labels[slice_idx] contains label values for that slice.
    affine : ndarray
        4x4 affine transformation matrix.
    """
    data = nib.load(nifti_file)
    labeled_image = data.get_fdata()
    affine = data.affine
    
    voxel_coords, labels = extract_label_coordinates(labeled_image, mode='contour')
    world_coords = transform_to_world_coordinates(voxel_coords, affine)
    
    return world_coords, labels, affine

def load_planes(nifti_file):
    """
    Load planes (all voxels) from a NIfTI file with geometry information.
    
    Parameters:
    -----------
    nifti_file : str
        Path to NIfTI file (SAX or LAX).
    
    Returns:
    --------
    planes : list of lists
        planes[slice_idx][label_idx] contains xyz coordinates (3, n_points).
    labels : list of lists
        labels[slice_idx] contains label values for that slice.
    ipp : ndarray or list
        Image plane position(s).
    ipo : ndarray
        Image plane orientation (6,1).
    pxs : list
        Pixel spacing [row_spacing, col_spacing].
    labeled_image : ndarray
        Original labeled image data.
    """
    data = nib.load(nifti_file)
    labeled_image = data.get_fdata()
    affine = data.affine
    
    voxel_coords, labels = extract_label_coordinates(labeled_image, mode='plane')
    world_coords = transform_to_world_coordinates(voxel_coords, affine)
    ipp, ipo, pxs = compute_plane_geometry(affine, labeled_image)
    
    return world_coords, labels, ipp, ipo, pxs, labeled_image

# ============================================================================
# DATA LOADING - BACKWARD COMPATIBILITY WRAPPERS (optional - for easier migration)
# ============================================================================

def load_sax_pc(sax_file):
    """Load SAX contours. Returns (sax_pcs, lab_sax_lb, affine_sax)."""
    return load_contours(sax_file)

def load_lax_pc(ch2_file):
    """Load LAX contours. Returns (lax_pc, affine_lax)."""
    contours, labels, affine = load_contours(ch2_file)
    # Flatten for single slice to match original behavior
    return contours[0], affine

def load_sax_ps(sax_file):
    """Load SAX planes. Returns (sax_pcs, lab_sax_lb, ipp, ipo, pxs, lab_sax)."""
    return load_planes(sax_file)

def load_lax_ps(ch2_file):
    """Load LAX planes. Returns (lax_pc, ipp, ipo, pxs, lab_lax)."""
    planes, labels, ipp, ipo, pxs, labeled_image = load_planes(ch2_file)
    # Flatten for single slice to match original behavior
    return planes[0], ipp, ipo, pxs, labeled_image

def rand_tri(plane_coords):
    """
    Compute plane normal from three random points using maximum distance criterion.
    
    Parameters:
    -----------
    plane_coords : list
        plane_coords[0] contains coordinates (3, n_points) for the first label.
    
    Returns:
    --------
    n : ndarray
        Unit normal vector (3,) perpendicular to the plane.
    """
    points = np.transpose(plane_coords[0])
    
    # Pick random starting point
    i0 = random.choice(list(range(len(points))))
    x0 = points[i0]
    
    # Find farthest point from x0
    d0 = np.linalg.norm(points - x0, axis=1)
    i1 = np.argmax(d0)
    x1 = points[i1]
    
    # Find point that maximizes sum of distances
    d1 = np.linalg.norm(points - x1, axis=1)
    i2 = np.argmax(d0 + d1)
    x2 = points[i2]
    
    # Compute and normalize cross product
    n = np.cross(x2 - x0, x1 - x0)
    n = n / np.linalg.norm(n)
    
    return n

def compute_cardiac_axes(ch2_ps, ch4_ps):
    """
    Compute cardiac coordinate system axes from 2CH and 4CH plane data.
    
    Parameters:
    -----------
    ch2_ps : list of ndarrays
        2CH plane coordinates. ch2_ps[1] = LV, ch2_ps[3] = LA.
    ch4_ps : list of ndarrays
        4CH plane coordinates. ch4_ps[1] = LV, ch4_ps[3] = RV.
    
    Returns:
    --------
    ax_ab : ndarray
        Apex-to-base axis (3,), pointing from LV to LA.
    ax_lr : ndarray
        Left-to-right axis (3,), pointing from LV to RV.
    ax_fb : ndarray
        Front-to-back axis (3,), perpendicular to ax_ab and ax_lr.
    """
    # Compute plane normals
    n_2ch = rand_tri(ch2_ps)
    n_4ch = rand_tri(ch4_ps)
    
    # Apex-to-base axis (cross product of plane normals)
    ax_l0 = np.cross(n_2ch, n_4ch)
    ax_l0 = ax_l0 / np.linalg.norm(ax_l0)
    
    # Orient apex-to-base: LV → LA
    lv_2ch = np.mean(ch2_ps[1], axis=1)
    la_2ch = np.mean(ch2_ps[3], axis=1)
    ax_ab = ax_l0 if np.dot(la_2ch - lv_2ch, ax_l0) > 0 else -ax_l0
    
    # Left-to-right axis (perpendicular to 4CH plane and apex-base)
    ax_r0 = np.cross(n_4ch, ax_ab)
    ax_r0 = ax_r0 / np.linalg.norm(ax_r0)
    
    # Orient left-to-right: LV → RV
    lv_4ch = np.mean(ch4_ps[1], axis=1)
    rv_4ch = np.mean(ch4_ps[3], axis=1)
    ax_lr = ax_r0 if np.dot(rv_4ch - lv_4ch, ax_r0) > 0 else -ax_r0
    
    # Front-to-back axis (perpendicular to both)
    ax_fb = np.cross(ax_ab, ax_lr)
    ax_fb = ax_fb / np.linalg.norm(ax_fb)
    
    return ax_ab, ax_lr, ax_fb

def compute_volume_grid(ch2_pc, ch4_pc, ax_ab, ax_lr, ax_fb, grid_size=160, margin_factor=1.2):
    """
    Create a 3D volume grid aligned with cardiac axes, centered on LAX contours.
    
    Parameters:
    -----------
    ch2_pc : list
        2CH contour coordinates.
    ch4_pc : list
        4CH contour coordinates.
    ax_ab, ax_lr, ax_fb : ndarray
        Cardiac axis vectors (3,).
    grid_size : int
        Volume grid dimension (default: 160).
    margin_factor : float
        Margin around data (default: 1.2 = 20% margin).
    
    Returns:
    --------
    xyz_v : ndarray
        World coordinates for all grid points (grid_size^3, 3).
    ijk_v : ndarray
        Grid indices centered at 0 (grid_size^3, 3).
    o_c_ : ndarray
        Grid origin in world coordinates (3,).
    vs : float
        Voxel spacing.
    vm : ndarray
        Rotation matrix (3, 3) mapping grid to world coordinates.
    affine_3d : ndarray
        4x4 affine transformation matrix.
    """
    # Collect all LAX contour points
    cs = []
    for k2 in ch2_pc:
        for kl in range(np.shape(k2)[1]):
            cs.append(np.array(k2)[:, kl])
    for k4 in ch4_pc:
        for kl in range(np.shape(k4)[1]):
            cs.append(np.array(k4)[:, kl])
    cs = np.array(cs)
    
    # Compute bounding box center
    min_lax = np.min(cs, axis=0)
    max_lax = np.max(cs, axis=0)
    o_c = np.mean((min_lax, max_lax), axis=0)
    
    # Project onto each axis to find extents
    d_ab = np.dot(cs - o_c, ax_ab).max() - np.dot(cs - o_c, ax_ab).min()
    d_lr = np.dot(cs - o_c, ax_lr).max() - np.dot(cs - o_c, ax_lr).min()
    d_fb = np.dot(cs - o_c, ax_fb).max() - np.dot(cs - o_c, ax_fb).min()
    
    # Center along each axis
    c_ab = (np.dot(cs - o_c, ax_ab).max() + np.dot(cs - o_c, ax_ab).min()) / 2
    c_lr = (np.dot(cs - o_c, ax_lr).max() + np.dot(cs - o_c, ax_lr).min()) / 2
    c_fb = (np.dot(cs - o_c, ax_fb).max() + np.dot(cs - o_c, ax_fb).min()) / 2
    o_c_ = o_c + c_ab * ax_ab + c_lr * ax_lr + c_fb * ax_fb
    
    # Compute voxel spacing with margin
    vs = np.max([d_ab, d_lr, d_fb]) * margin_factor / grid_size
    
    # Create grid indices (centered at 0)
    ii = np.linspace(0, grid_size - 1, grid_size) - grid_size / 2
    jj = np.linspace(0, grid_size - 1, grid_size) - grid_size / 2
    kk = np.linspace(0, grid_size - 1, grid_size) - grid_size / 2
    iv, jv, kv = np.meshgrid(ii, jj, kk)
    ijk_v = np.array([np.resize(iv, np.size(iv)), 
                      np.resize(jv, np.size(jv)), 
                      np.resize(kv, np.size(kv))]).transpose()
    
    # Rotation matrix: grid axes → world axes
    vm = np.array([ax_ab, ax_lr, ax_fb]).transpose()
    
    # Transform grid to world coordinates
    xyz_v = o_c_ + np.dot(vm, ijk_v.transpose()).transpose() * vs
    
    # Create affine matrix
    affine_3d = np.eye(4)
    affine_3d[0:3, 0:3] = vm * vs
    affine_3d[0:3, 3] = xyz_v[0, :]
    
    return xyz_v, ijk_v, o_c_, vs, vm, affine_3d

def project_slice_to_volume(xyz_v, ijk_v, vs, plane_normal, plane_point, 
                            slice_ipp, slice_ipo, slice_pxs, slice_lab, 
                            label_mapping, grid_size=160):
    """
    Project a 2D slice onto the 3D volume grid (forward projection).
    
    Parameters:
    -----------
    xyz_v : ndarray
        World coordinates of volume grid (n_voxels, 3).
    ijk_v : ndarray
        Grid indices (n_voxels, 3).
    vs : float
        Voxel spacing.
    plane_normal : ndarray
        Unit normal to the slice plane (3,).
    plane_point : ndarray
        A point on the slice plane (3,).
    slice_ipp : ndarray
        Image plane position (3,).
    slice_ipo : ndarray
        Image plane orientation (6,) - row and column direction vectors.
    slice_pxs : list
        Pixel spacing [row, col].
    slice_lab : ndarray
        Slice label image.
    label_mapping : list or dict
        Maps slice labels to volume labels.
    grid_size : int
        Volume grid dimension (default: 160).
    
    Returns:
    --------
    vol_sp : ndarray
        3D volume with projected labels (grid_size, grid_size, grid_size).
    """
    vol_sp = np.zeros((grid_size, grid_size, grid_size))
    ijk_v_ = ijk_v + grid_size // 2  # Shift to positive indices
    
    # Find voxels near the plane
    d = np.dot(xyz_v - plane_point, plane_normal)
    i_slice = np.where(np.abs(d) <= vs)[0]
    xyz_slice = xyz_v[i_slice, :]
    
    # Project to slice pixel coordinates
    v0_s = slice_ipo[0:3]
    v1_s = slice_ipo[3:]
    pq_slice = np.transpose([
        np.dot(xyz_slice - np.transpose(slice_ipp), v0_s) / slice_pxs[0],
        np.dot(xyz_slice - np.transpose(slice_ipp), v1_s) / slice_pxs[1]
    ]).squeeze().round()
    
    # Transfer labels from slice to volume
    for ki in range(len(i_slice)):
        try:
            p, q = pq_slice[ki][0].astype(int), pq_slice[ki][1].astype(int)
            slice_idx = 0 if len(slice_lab.shape) == 2 else slice_lab.shape[2] - 1
            
            if len(slice_lab.shape) == 3:
                label = slice_lab[p, q, slice_idx].astype(int)
            else:
                label = slice_lab[p, q].astype(int)
            
            if isinstance(label_mapping, dict):
                vol_label = label_mapping.get(label, 0)
            else:
                vol_label = label_mapping[label] if label < len(label_mapping) else 0
            
            if vol_label > 0 or len(label_mapping) > 4:  # Keep all labels for plane mode
                i, j, k = ijk_v_[i_slice[ki]].astype(int)
                vol_sp[i, j, k] = vol_label
        except (IndexError, KeyError):
            print('Out of box voxels.', end='\r')
    
    return vol_sp

def project_volume_to_slice(xyz_v, ijk_v, vs, plane_normal, plane_point,
                            slice_ipp, slice_ipo, slice_pxs, slice_shape,
                            vol_ds, label_mapping, grid_size=160):
    """
    Project 3D volume onto a 2D slice (back projection).
    
    Parameters:
    -----------
    xyz_v : ndarray
        World coordinates of volume grid (n_voxels, 3).
    ijk_v : ndarray
        Grid indices (n_voxels, 3).
    vs : float
        Voxel spacing.
    plane_normal : ndarray
        Unit normal to the slice plane (3,).
    plane_point : ndarray
        A point on the slice plane (3,).
    slice_ipp : ndarray
        Image plane position (3,).
    slice_ipo : ndarray
        Image plane orientation (6,).
    slice_pxs : list
        Pixel spacing [row, col].
    slice_shape : tuple
        Shape of output slice (rows, cols, slices).
    vol_ds : ndarray
        3D volume with labels (grid_size, grid_size, grid_size).
    label_mapping : list or dict
        Maps volume labels to slice labels.
    grid_size : int
        Volume grid dimension (default: 160).
    
    Returns:
    --------
    slice_bp : ndarray
        2D slice with back-projected labels.
    """
    slice_bp = np.zeros(slice_shape)
    ijk_v_ = ijk_v + grid_size // 2
    
    # Find voxels near the plane
    d = np.dot(xyz_v - plane_point, plane_normal)
    i_slice = np.where(np.abs(d) <= vs)[0]
    xyz_slice = xyz_v[i_slice, :]
    
    # Project to slice pixel coordinates
    v0_s = slice_ipo[0:3]
    v1_s = slice_ipo[3:]
    pq_slice = np.transpose([
        np.dot(xyz_slice - np.transpose(slice_ipp), v0_s) / slice_pxs[0],
        np.dot(xyz_slice - np.transpose(slice_ipp), v1_s) / slice_pxs[1]
    ]).squeeze().round()
    
    # Transfer labels from volume to slice
    for ki in range(len(i_slice)):
        try:
            p, q = pq_slice[ki][0].astype(int), pq_slice[ki][1].astype(int)
            i, j, k = ijk_v_[i_slice[ki]].astype(int)
            vol_label = vol_ds[i, j, k].astype(int)
            
            if isinstance(label_mapping, dict):
                slice_label = label_mapping.get(vol_label, 0)
            else:
                slice_label = label_mapping[vol_label] if vol_label < len(label_mapping) else 0
            
            if len(slice_shape) == 3:
                slice_bp[p, q, 0] = slice_label
            else:
                slice_bp[p, q] = slice_label
        except (IndexError, KeyError):
            print('Out of box voxels.', end='\r')
    
    return slice_bp


# ============================================================================
# HIGH-LEVEL API FUNCTIONS - 3D RECONSTRUCTION AND BACK-PROJECTION
# ============================================================================

def vol_grid_gen(ch2_ps, ch4_ps, sax_ps,
                 ch2_pc, ch4_pc, sax_pc,
                 sax_ipp, sax_ipo, sax_pxs, sax_lab,
                 ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                 ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab):
    """
    Generate 3D volume from 2D slices (2CH, 4CH, SAX).
    
    Returns:
    --------
    vol_sp : ndarray
        3D volume with labels (160, 160, 160).
    affine_3d : ndarray
        4x4 affine transformation matrix.
    """
    # Compute cardiac coordinate system
    ax_ab, ax_lr, ax_fb = compute_cardiac_axes(ch2_ps, ch4_ps)
    
    # Create volume grid
    xyz_v, ijk_v, o_c_, vs, vm, affine_3d = compute_volume_grid(
        ch2_pc, ch4_pc, ax_ab, ax_lr, ax_fb
    )
    
    # Initialize volume
    vol_sp = np.zeros((160, 160, 160))
    
    # Project 2CH
    n_2ch = rand_tri(ch2_ps)
    vol_2ch = project_slice_to_volume(
        xyz_v, ijk_v, vs, n_2ch, ch2_pc[0][:, 0],
        ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
        label_mapping=[0, 1, 2, 5]  # Maps slice labels to volume labels
    )
    vol_sp = np.maximum(vol_sp, vol_2ch)
    
    # Project 4CH
    n_4ch = rand_tri(ch4_ps)
    vol_4ch = project_slice_to_volume(
        xyz_v, ijk_v, vs, n_4ch, ch4_pc[0][:, 0],
        ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
        label_mapping=[0, 1, 2, 3, 5, 6]
    )
    vol_sp = np.maximum(vol_sp, vol_4ch)
    
    # Project SAX slices
    ns = len(sax_pc)
    for ks in range(ns):
        sax_ps_ks = sax_ps[ks]
        n_ks = rand_tri(sax_ps_ks)
        
        # Extract slice at index ks
        sax_lab_ks = sax_lab[..., ks] if len(sax_lab.shape) == 3 else sax_lab
        
        vol_sax_ks = project_slice_to_volume(
            xyz_v, ijk_v, vs, n_ks, sax_ps_ks[0][:, 0],
            sax_ipp[ks], sax_ipo, sax_pxs, sax_lab,
            label_mapping=[0, 1, 2, 3]
        )
        # Only overwrite with non-zero labels from SAX
        mask = vol_sax_ks > 0
        vol_sp[mask] = vol_sax_ks[mask]
    
    return vol_sp, affine_3d

def vol_grid_bp(ch2_ps, ch4_ps, sax_ps,
                ch2_pc, ch4_pc, sax_pc,
                sax_ipp, sax_ipo, sax_pxs, sax_lab,
                ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
                vol_pr):
    """
    Back-project 3D volume to 2D slices (2CH, 4CH, SAX).
    
    Parameters:
    -----------
    vol_pr : ndarray
        3D volume with predicted labels (160, 160, 160).
    
    Returns:
    --------
    ch2_bp : ndarray
        2CH back-projected labels.
    ch4_bp : ndarray
        4CH back-projected labels.
    sax_bp : ndarray
        SAX back-projected labels.
    """
    # Compute cardiac coordinate system
    ax_ab, ax_lr, ax_fb = compute_cardiac_axes(ch2_ps, ch4_ps)
    
    # Create volume grid
    xyz_v, ijk_v, o_c_, vs, vm, affine_3d = compute_volume_grid(
        ch2_pc, ch4_pc, ax_ab, ax_lr, ax_fb
    )
    
    # Back-project to 2CH
    n_2ch = rand_tri(ch2_ps)
    ch2_bp = project_volume_to_slice(
        xyz_v, ijk_v, vs, n_2ch, ch2_pc[0][:, 0],
        ch2_ipp, ch2_ipo, ch2_pxs, np.shape(ch2_lab),
        vol_pr, label_mapping=[0, 1, 2, 0, 0, 3, 0, 0, 0]
    )
    
    # Back-project to 4CH
    n_4ch = rand_tri(ch4_ps)
    ch4_bp = project_volume_to_slice(
        xyz_v, ijk_v, vs, n_4ch, ch4_pc[0][:, 0],
        ch4_ipp, ch4_ipo, ch4_pxs, np.shape(ch4_lab),
        vol_pr, label_mapping=[0, 1, 2, 3, 0, 4, 5, 0, 0]
    )
    
    # Back-project to SAX slices
    ns = len(sax_pc)
    sax_bp = np.zeros(np.shape(sax_lab))
    for ks in range(ns):
        sax_ps_ks = sax_ps[ks]
        n_ks = rand_tri(sax_ps_ks)
        
        sax_bp_ks = project_volume_to_slice(
            xyz_v, ijk_v, vs, n_ks, sax_ps_ks[0][:, 0],
            sax_ipp[ks], sax_ipo, sax_pxs, (sax_lab.shape[0], sax_lab.shape[1], 1),
            vol_pr, label_mapping=[0, 1, 2, 3, 0, 0, 0, 0, 0]
        )
        sax_bp[..., ks] = sax_bp_ks[..., 0]
    
    return ch2_bp, ch4_bp, sax_bp

def load_model(model_file: str, device_str: str = 'cpu') -> nn.Module: 
    """
    Load the 3D reconstruction model from checkpoint.
    """
    if device_str not in ['cpu', 'cuda']:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model checkpoint not found at {model_file}")    
    
    # Loading model
    print(f'Loading model on device: {device_str}')
    device = torch.device(device_str) 
    is_cpu = device.type == 'cpu'
    unet = ReconstructUNet3D(in_channel=1, out_channel=9)
    checkpoint = torch.load(model_file)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device, dtype=torch.float)
    if not is_cpu:
        unet.cuda()
    else:
        unet.cpu()
    
    return unet

def run_3d_reconstruction(model: nn.Module, sax_file: str, ch2_file: str, ch4_file:str, output_dir: str, subject_id: str, device_str:str = 'cpu', compute_bp = True) -> dict : 
    """
    Run 3D reconstruction from SAX, 2CH, and 4CH NIfTI files.
    """
    if device_str not in ['cpu', 'cuda']:
        raise ValueError("Device must be 'cpu' or 'cuda'")

    outputs = ReconstructOutputManager(base_output_dir=output_dir, subject_id=subject_id)
    
    # Loading model
    unet = model
    unet.eval()

    sax_pc, sax_lb_1, sax_affine = load_contours(sax_file)
    sax_ps, sax_lb_0, sax_ipp, sax_ipo, sax_pxs, sax_lab = load_planes(sax_file)

    ch2_pc, _, ch2_affine = load_contours(ch2_file)
    ch4_pc, _, ch4_affine = load_contours(ch4_file)
    ch2_pc = ch2_pc[0]  # Flatten for single slice
    ch4_pc = ch4_pc[0]  # Flatten for single slice

    ch2_ps, _, ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab = load_planes(ch2_file)
    ch4_ps, _, ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab = load_planes(ch4_file)
    ch2_ps = ch2_ps[0]  # Flatten for single slice
    ch4_ps = ch4_ps[0]  # Flatten for single slice

    # sparse volume generation
    vol_sp, affine_3d = vol_grid_gen(ch2_ps, ch4_ps, sax_ps, 
                                     ch2_pc, ch4_pc, sax_pc, 
                                     sax_ipp, sax_ipo, sax_pxs, sax_lab, 
                                     ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab, 
                                     ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab)
    
    vol_sp_nif = nib.Nifti1Image(vol_sp, affine=affine_3d)
    nib.save(vol_sp_nif, outputs.get_path('sparse_volume'))

    # network prediction
    img_i_ = np.transpose(vol_sp, [1, 0, 2])
    test_x = img_i_[np.newaxis, np.newaxis, ...] * 30
    tst_x = Variable(torch.from_numpy(test_x).float().to(unet.device))
    output = unet(tst_x)
    prd = output.cpu().detach().numpy()
    
    del test_x, tst_x, output

    if device_str == 'cuda':
        torch.cuda.empty_cache()

    lab = np.argmax(prd, axis=1)[0, ...]
    lab_ = np.transpose(lab, [1, 0, 2])
    prd_nif = nib.Nifti1Image(lab_.astype(float), affine=affine_3d)
    nib.save(prd_nif, outputs.get_path('prediction'))

    output_dict = outputs.get_all_paths()
    # back projection 2d
    if compute_bp:
        ch2_bp, ch4_bp, sax_bp = vol_grid_bp(ch2_ps, ch4_ps, sax_ps,
                                         ch2_pc, ch4_pc, sax_pc,
                                         sax_ipp, sax_ipo, sax_pxs, sax_lab,
                                         ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                                         ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
                                         lab_)
        ch2_bp_nif = nib.Nifti1Image(ch2_bp, affine=ch2_affine)
        nib.save(ch2_bp_nif, outputs.get_path('ch2_bp'))
        ch4_bp_nif = nib.Nifti1Image(ch4_bp, affine=ch4_affine)
        nib.save(ch4_bp_nif, outputs.get_path('ch4_bp'))
        sax_bp_nif = nib.Nifti1Image(sax_bp, affine=sax_affine)
        nib.save(sax_bp_nif, outputs.get_path('sax_bp'))        
    else : 
        output_dict['ch2_bp'] = ''
        output_dict['ch4_bp'] = ''
        output_dict['sax_bp'] = ''
    
    return output_dict


def run_3d_reconstruction(sax_file, ch2_file, ch4_file, save_name_sp, save_name_prd,
                    save_name_sax_bp, save_name_ch2_bp, save_name_ch4_bp,model_dir=None):
    if model_dir is None:
        raise ValueError("model_dir argument must be provided")

    # Build full model path assuming checkpoint filename is fixed
    model_file = os.path.join(model_dir, 'epoch_150_params.pth')

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model checkpoint not found at {model_file}")
    
    #model_file = '/exp_0/epoch_150_params.pth'
    
    # device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device_name}")
    # device = torch.device(device_name)
    device = torch.device('cpu')

    print(f'Loading model on device: {device}')

    unet = ReconstructUNet3D(in_channel=1, out_channel=9)
    checkpoint = torch.load(model_file)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device, dtype=torch.float)
    if device.type == 'cuda':
        unet.cuda()
    else:
        unet.cpu()
    # unet.cuda()

    sax_pc, sax_lb_1, sax_affine = load_sax_pc(sax_file)
    sax_ps, sax_lb_0, sax_ipp, sax_ipo, sax_pxs, sax_lab = load_sax_ps(sax_file)
    ch2_pc, ch2_affine = load_lax_pc(ch2_file)
    ch4_pc, ch4_affine = load_lax_pc(ch4_file)
    ch2_ps, ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab = load_lax_ps(ch2_file)
    ch4_ps, ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab = load_lax_ps(ch4_file)
    vol_sp, affine_3d = vol_grid_gen(ch2_ps, ch4_ps, sax_ps,
                                     ch2_pc, ch4_pc, sax_pc,
                                     sax_ipp, sax_ipo, sax_pxs, sax_lab,
                                     ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                                     ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab)
    vol_sp_nif = nib.Nifti1Image(vol_sp, affine=affine_3d)
    nib.save(vol_sp_nif, save_name_sp)

    # network prediction
    img_i_ = np.transpose(vol_sp, [1, 0, 2])
    test_x = img_i_[np.newaxis, np.newaxis, ...] * 30
    if device.type == 'cuda':
        tst_x = Variable(torch.from_numpy(test_x).float().cuda())
    else:
        tst_x = Variable(torch.from_numpy(test_x).float().cpu())
    output = unet(tst_x)
    prd = output.cpu().detach().numpy()
    del test_x, tst_x, output
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    lab = np.argmax(prd, axis=1)[0, ...]
    lab_ = np.transpose(lab, [1, 0, 2])
    prd_nif = nib.Nifti1Image(lab_.astype(float), affine=affine_3d)
    nib.save(prd_nif, save_name_prd)

    # back projection 2d
    ch2_bp, ch4_bp, sax_bp = vol_grid_bp(ch2_ps, ch4_ps, sax_ps,
                                         ch2_pc, ch4_pc, sax_pc,
                                         sax_ipp, sax_ipo, sax_pxs, sax_lab,
                                         ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                                         ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
                                         lab_)
    ch2_bp_nif = nib.Nifti1Image(ch2_bp, affine=ch2_affine)
    nib.save(ch2_bp_nif, save_name_ch2_bp)
    ch4_bp_nif = nib.Nifti1Image(ch4_bp, affine=ch4_affine)
    nib.save(ch4_bp_nif, save_name_ch4_bp)
    sax_bp_nif = nib.Nifti1Image(sax_bp, affine=sax_affine)
    nib.save(sax_bp_nif, save_name_sax_bp)
    return