import os

import numpy as np
import nibabel as nib

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import cardio_form.geometry as geometry
import cardio_form.io as cf_io
from cardio_form.utils import configure_logging
logger = configure_logging('Reconstruct3D')
from cardio_form.output_managers import OutputManager

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
    
def load_model(model_file: str, device_str: str = 'cpu') -> nn.Module: 
    """
    Load the 3D reconstruction model from checkpoint.
    """
    if device_str not in ['cpu', 'cuda']:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model checkpoint not found at {model_file}")    
    
    # Loading model
    logger.info(f'Loading model on device: {device_str}')
    device = torch.device(device_str) 
    is_cpu = device.type == 'cpu'
    
    unet = ReconstructUNet3D(in_channel=1, out_channel=9)
    checkpoint = torch.load(model_file, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device, dtype=torch.float)
    if not is_cpu:
        unet.cuda()
    else:
        unet.cpu()
    
    return unet

def run_3d_reconstruction(model, sax_file, ch2_file, ch4_file, output_dir, 
                         output_prefix, device_str='cpu', compute_qc=False):
    """
    Run 3D reconstruction from SAX, 2CH, and 4CH NIfTI files.
    
    This function orchestrates the complete reconstruction pipeline:
    1. Loads 2D segmentation data from input NIfTI files
    2. Projects 2D slices into a sparse 3D volume (oblique coordinate system)
    3. Runs 3D U-Net to densify the segmentation
    4. Optionally back-projects the 3D result to 2D slices for validation
    
    Parameters:
    -----------
    model : nn.Module
        Pre-loaded 3D U-Net model
    sax_file : str
        Path to short-axis NIfTI file
    ch2_file : str
        Path to 2-chamber long-axis NIfTI file
    ch4_file : str
        Path to 4-chamber long-axis NIfTI file
    output_dir : str
        Directory for output files
    output_prefix : str
        Prefix for all output filenames (e.g., 'subject_001')
    device_str : str, optional
        'cpu' or 'cuda' (default: 'cpu')
    compute_qc : bool, optional
        Whether to write diagnostic artefacts: the sparse volume and the three
        back-projections (default: False)
    
    Returns:
    --------
    dict
        Dictionary containing paths to all output files
    
    Notes:
    ------
    - The output uses oblique coordinates (the reconstruction's native grid)
    - Back-projections use the same oblique geometry, for consistency
    - Diagnostic artefacts are only written when compute_qc is True
    """
    
    if device_str not in ['cpu', 'cuda']:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    
    outputs = OutputManager(output_dir=output_dir, output_prefix=output_prefix)
    
    # Set model to evaluation mode
    model.eval()
    
    # ========================================================================
    # STEP 1: Load all input data
    # ========================================================================
    logger.info("Loading input segmentations...")
    
    # Load each NIfTI once; geometry now operates on arrays + affines.
    sax_data, sax_affine = cf_io.load_data_and_affine(sax_file)
    ch2_data, ch2_affine = cf_io.load_data_and_affine(ch2_file)
    ch4_data, ch4_affine = cf_io.load_data_and_affine(ch4_file)

    # SAX (multi-slice short-axis)
    sax_pc, _, _ = geometry.compute_sax_contour_geometry(sax_data, sax_affine)
    sax_ps, _, sax_ipp, sax_ipo, sax_pxs, sax_lab = geometry.compute_sax_plane_geometry(sax_data, sax_affine)

    # 2- and 4-chamber long-axis
    ch2_pc, _ = geometry.compute_lax_contour_geometry(ch2_data, ch2_affine)
    ch4_pc, _ = geometry.compute_lax_contour_geometry(ch4_data, ch4_affine)
    ch2_ps, ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab = geometry.compute_lax_plane_geometry(ch2_data, ch2_affine)
    ch4_ps, ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab = geometry.compute_lax_plane_geometry(ch4_data, ch4_affine)
     
    # ========================================================================
    # STEP 2: Generate sparse 3D volume (forward projection)
    # ========================================================================
    logger.info("Generating sparse 3D volume...")
    
    vol_sp, affine_3d = geometry.vol_grid_gen(
        ch2_ps, ch4_ps, sax_ps, 
        ch2_pc, ch4_pc, sax_pc, 
        sax_ipp, sax_ipo, sax_pxs, sax_lab, 
        ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab, 
        ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab
    )
    
    # Check coordinate system handedness
    det = np.linalg.det(affine_3d[:3, :3])
    if det < 0:
        logger.info("WARNING: Generated affine has negative determinant (left-handed system)")
    else:
        logger.info(f"Affine determinant: {det:.4f} (right-handed system)")
    
    # Save sparse volume in original oblique coordinates
    if compute_qc:
        vol_sp_nif = nib.Nifti1Image(vol_sp, affine=affine_3d)
        nib.save(vol_sp_nif, outputs.get_path('sparse_volume'))
    
    # ========================================================================
    # STEP 3: Run 3D U-Net prediction
    # ========================================================================
    logger.info("Running 3D U-Net prediction...")
    
    # Prepare input tensor
    img_transposed = np.transpose(vol_sp, [1, 0, 2])
    test_x = img_transposed[np.newaxis, np.newaxis, ...] * 30
    
    # Move to appropriate device
    if device_str == 'cuda':
        tst_x = Variable(torch.from_numpy(test_x).float().cuda())
    else:
        tst_x = Variable(torch.from_numpy(test_x).float().cpu())
    
    # Forward pass
    with torch.no_grad():
        output = model(tst_x)
    
    # Extract prediction and clean up
    prd = output.cpu().detach().numpy()
    del test_x, tst_x, output
    
    if device_str == 'cuda':
        torch.cuda.empty_cache()
    
    # Convert probabilities to labels
    lab = np.argmax(prd, axis=1)[0, ...]
    lab_transposed = np.transpose(lab, [1, 0, 2])
    
    # Create oblique prediction image
    oblique_nii = nib.Nifti1Image(lab_transposed.astype(np.uint8), affine=affine_3d)
    
    # ========================================================================
    # STEP 4: Save the segmentation
    # ========================================================================
    path_oblique = outputs.get_path('prediction')
    nib.save(oblique_nii, path_oblique)
    logger.info(f"Saved whole-heart segmentation: {path_oblique}")

    output_dict = {'prediction': path_oblique}
    if compute_qc:
        output_dict['sparse_volume'] = outputs.get_path('sparse_volume')

    # ========================================================================
    # STEP 5: Back-projection (optional, uses oblique geometry)
    # ========================================================================
    if compute_qc:
        logger.info("Computing back-projections...")
        
        # Back-project using OBLIQUE geometry (not LPS!)
        # This ensures geometric consistency with the original slice positions
        ch2_bp, ch4_bp, sax_bp = geometry.vol_grid_bp(
            ch2_ps, ch4_ps, sax_ps,
            ch2_pc, ch4_pc, sax_pc,
            sax_ipp, sax_ipo, sax_pxs, sax_lab,
            ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
            ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
            lab_transposed
        )
        
        # Save back-projections
        nib.save(nib.Nifti1Image(ch2_bp, affine=ch2_affine), 
                outputs.get_path('ch2_bp'))
        nib.save(nib.Nifti1Image(ch4_bp, affine=ch4_affine), 
                outputs.get_path('ch4_bp'))
        nib.save(nib.Nifti1Image(sax_bp, affine=sax_affine), 
                outputs.get_path('sax_bp'))

        for key in ('ch2_bp', 'ch4_bp', 'sax_bp'):
            output_dict[key] = outputs.get_path(key)
    
    logger.info("Reconstruction complete!")
    return output_dict