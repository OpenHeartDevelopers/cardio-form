import os
import numpy as np
import nibabel as nib
from scipy.interpolate import NearestNDInterpolator
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from cardio_form.output_managers import OutputManager
from cardio_form.utils import configure_logging
logger = configure_logging(__name__)

# network
class ReconstructLaUNet3d(nn.Module):
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
        super(ReconstructLaUNet3d, self).__init__()
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

# prepare sparse volume
def ijk_vol(img):
    sz = np.shape(img)
    idx = []
    value = []
    for i in range(sz[0]):
        for j in range(sz[1]):
            for k in range(sz[2]):
                idx.append(np.array([[i], [j], [k], [1]]))
                value.append(img[i, j, k])
    return idx, value


def image_points_nifti(img_file):
    img_data = nib.load(img_file)
    img = img_data.get_fdata()
    img_data = nib.load(img_file)
    affine = img_data.affine
    idx, value = ijk_vol(img)
    xs = [np.matmul(affine, x) for x in idx]
    return [x[0:3] for x in idx], [x[0:3] for x in xs], value, img, affine


def find_points(i_k, i_t):
    xt = np.where(np.array(i_k)[:, 0, 0] == i_t[0])
    yt = np.where(np.array(i_k)[:, 1, 0] == i_t[1])
    zt = np.where(np.array(i_k)[:, 2, 0] == i_t[2])
    ii = [x for x in xt[0] if x in yt[0]]
    i = [x for x in ii if x in zt[0]]
    return i[0]


def plaine_info(i_k, x_k):
    i0 = find_points(i_k, [[0], [0], [0]])
    i1 = find_points(i_k, [[1], [0], [0]])
    i2 = find_points(i_k, [[0], [1], [0]])
    v1 = np.squeeze(x_k[i1] - x_k[i0])
    v2 = np.squeeze(x_k[i2] - x_k[i0])
    p = np.squeeze(x_k[i0])
    n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    return p, n


def lax_coords(i_2ch, x_2ch, v_2ch, i_4ch, x_4ch, v_4ch):
    p_2ch, n_2ch_ = plaine_info(i_2ch, x_2ch)
    p_4ch, n_4ch_ = plaine_info(i_4ch, x_4ch)

    d_2ch = [np.abs(np.dot(np.squeeze(x) - p_4ch, n_4ch_)) for x in x_2ch]
    lax_idx_2ch = np.where(np.array(d_2ch) <= 1)[0]
    x_lax_2ch = [x_2ch[x] for x in lax_idx_2ch]
    v_lax_2ch = [v_2ch[x] for x in lax_idx_2ch]
    la_idx_lax_2ch = np.where(np.array(v_lax_2ch) == 1)[0]
    la_c_lax_2ch = np.array([x_lax_2ch[x] for x in la_idx_lax_2ch]).mean(axis=0)

    d_4ch = [np.abs(np.dot(np.squeeze(x) - p_2ch, n_2ch_)) for x in x_4ch]
    lax_idx_4ch = np.where(np.array(d_4ch) <= 1)[0]
    x_lax_4ch = [x_4ch[x] for x in lax_idx_4ch]
    v_lax_4ch = [v_4ch[x] for x in lax_idx_4ch]
    la_idx_lax_4ch = np.where(np.array(v_lax_4ch) == 1)[0]
    la_c_lax_4ch = np.array([x_lax_4ch[x] for x in la_idx_lax_4ch]).mean(axis=0)

    la_c_lax = np.mean([la_c_lax_2ch, la_c_lax_4ch], axis=0).squeeze()

    # grid axis
    la_idx_4ch = np.where(np.array(v_4ch) == 1)[0]
    x_la_4ch = [x_4ch[x] for x in la_idx_4ch]
    la_c_4ch = np.squeeze(np.array(x_la_4ch).mean(axis=0))
    lv_idx_4ch = np.where(np.array(v_4ch) == 2)[0]
    x_lv_4ch = [x_4ch[x] for x in lv_idx_4ch]
    lv_c_4ch = np.squeeze(np.array(x_lv_4ch).mean(axis=0))
    rv_idx_4ch = np.where(np.array(v_4ch) == 4)[0]
    x_rv_4ch = [x_4ch[x] for x in rv_idx_4ch]
    rv_c_4ch = np.squeeze(np.array(x_rv_4ch).mean(axis=0))
    n_lax_ = np.cross(n_4ch_, n_2ch_)

    vc_4ch = np.cross(rv_c_4ch - lv_c_4ch, la_c_4ch - lv_c_4ch)
    n_4ch = vc_4ch / np.linalg.norm(vc_4ch)
    if np.dot(n_4ch, n_4ch_) < 0:
        n_4ch_i = n_4ch_ * (-1)
    else:
        n_4ch_i = n_4ch_

    vc_lax = la_c_4ch - lv_c_4ch
    vn_lax = vc_lax / np.linalg.norm(vc_lax)
    if np.dot(vn_lax, n_lax_) < 0:
        n_lax_i = n_lax_ * (-1)
    else:
        n_lax_i = n_lax_

    vc_2ch = np.cross(n_4ch_i, n_lax_i)
    n_2ch = vc_2ch / np.linalg.norm(vc_2ch)
    n_2ch_i = n_2ch
    return la_c_lax, [n_2ch_i, n_4ch_i, n_lax_i]


def grid_point_gen(c, o, n, v):
    xx = v[0]
    yx = v[1]
    zx = v[2]

    m = np.eye(4)
    m[0:3, 0:3] = np.transpose(v)
    m[0:3, 3] = o + (- c[0]) * xx + (- c[1]) * yx + (- c[2]) * zx

    kx = np.arange(0, n[0], 1)
    ky = np.arange(0, n[1], 1)
    kz = np.arange(0, n[2], 1)
    kkx, kky, kkz = np.meshgrid(kx, ky, kz)
    idx = np.concatenate((np.reshape(kkx, (np.size(kkx), 1)),
                          np.reshape(kky, (np.size(kky), 1)),
                          np.reshape(kkz, (np.size(kkz), 1))), axis=1)
    idx_p = np.concatenate((np.reshape(kkx, (np.size(kkx), 1)),
                            np.reshape(kky, (np.size(kky), 1)),
                            np.reshape(kkz, (np.size(kkz), 1)),
                            np.reshape(np.ones(np.shape(kkx)), (np.size(kkx), 1))), axis=1)
    points = np.transpose(np.matmul(m, np.transpose(idx_p))[0:3])
    return points, idx, m


def grid_point_slice(points, idx, i_k, x_k):
    p_k, n_k_ = plaine_info(i_k, x_k)
    d_k = [np.abs(np.dot(np.squeeze(x) - p_k, n_k_)) for x in points]
    idx_k = np.where(np.array(d_k) <= 1)[0]
    points_slice = [points[x] for x in idx_k]
    idx_slice = [idx[x] for x in idx_k]
    return points_slice, idx_slice


def grid_point_assign(vol_s, points_k, idx_k, interp_k):
    v_k = interp_k(np.squeeze(points_k)[:, 0], np.squeeze(points_k)[:, 1], np.squeeze(points_k)[:, 2])
    for k in range(len(v_k)):
        vol_s[idx_k[k][0], idx_k[k][1], idx_k[k][2]] = v_k[k]
    return vol_s


def image_load_nifti_ori(img_file_2ch, img_file_4ch):
    i_2ch, x_2ch, v_2ch, img_2ch, affine_2ch = image_points_nifti(img_file_2ch)
    interp_2ch = NearestNDInterpolator(list(zip(np.squeeze(x_2ch)[:, 0],
                                                np.squeeze(x_2ch)[:, 1],
                                                np.squeeze(x_2ch)[:, 2])),
                                       np.squeeze(v_2ch))
    i_4ch, x_4ch, v_4ch, img_4ch, affine_4ch = image_points_nifti(img_file_4ch)
    interp_4ch = NearestNDInterpolator(list(zip(np.squeeze(x_4ch)[:, 0],
                                                np.squeeze(x_4ch)[:, 1],
                                                np.squeeze(x_4ch)[:, 2])),
                                       np.squeeze(v_4ch))
    la_c_mean, v = lax_coords(i_2ch, x_2ch, v_2ch, i_4ch, x_4ch, v_4ch)
    c = [64, 64, 64]
    o = la_c_mean.squeeze()
    n = [128, 128, 128]
    points, idx, m = grid_point_gen(c, o, n, v)
    points_2ch, idx_2ch = grid_point_slice(points, idx, i_2ch, x_2ch)
    points_4ch, idx_4ch = grid_point_slice(points, idx, i_4ch, x_4ch)
    vol_s = np.zeros(n)
    vol_s = grid_point_assign(vol_s, points_2ch, idx_2ch, interp_2ch)
    vol_s = grid_point_assign(vol_s, points_4ch, idx_4ch, interp_4ch)
    nif_s = nib.Nifti1Image(vol_s, affine=m)
    return nif_s


def points_rm(o, v, i_2ch, x_2ch):
    i = np.dot((np.squeeze(x_2ch) - o), v[0])
    j = np.dot((np.squeeze(x_2ch) - o), v[1])
    k = np.dot((np.squeeze(x_2ch) - o), v[2])
    i_list = np.multiply(i < 63, i > -64)
    j_list = np.multiply(j < 63, j > -64)
    k_list = np.multiply(k < 63, k > -64)
    x_2ch_ = np.array(x_2ch)[np.multiply(np.multiply(i_list, j_list), k_list)]
    i_2ch_ = np.array(i_2ch)[np.multiply(np.multiply(i_list, j_list), k_list)]
    return i_2ch_, x_2ch_


def image_save_nifti_ori(img_file_2ch, img_file_4ch, prd_tyi):
    i_2ch, x_2ch, v_2ch, img_2ch, affine_2ch = image_points_nifti(img_file_2ch)
    i_4ch, x_4ch, v_4ch, img_4ch, affine_4ch = image_points_nifti(img_file_4ch)

    la_c_mean, v = lax_coords(i_2ch, x_2ch, v_2ch, i_4ch, x_4ch, v_4ch)
    c = [64, 64, 64]
    o = la_c_mean.squeeze()
    n = [128, 128, 128]
    points, idx, m = grid_point_gen(c, o, n, v)

    i_2ch_, x_2ch_ = points_rm(o, v, i_2ch, x_2ch)
    i_4ch_, x_4ch_ = points_rm(o, v, i_4ch, x_4ch)

    interp_vol = NearestNDInterpolator(list(zip(np.squeeze(points)[:, 0],
                                                np.squeeze(points)[:, 1],
                                                np.squeeze(points)[:, 2])),
                                       [prd_tyi[x[0], x[1], x[2]] for x in idx])

    vol_2ch = np.zeros(np.shape(img_2ch))
    vol_4ch = np.zeros(np.shape(img_4ch))
    vol_2ch = grid_point_assign(vol_2ch, x_2ch_, i_2ch_, interp_vol)
    vol_4ch = grid_point_assign(vol_4ch, x_4ch_, i_4ch_, interp_vol)
    nif_prd_bp_2ch = nib.Nifti1Image(vol_2ch.astype(np.uint8), affine=affine_2ch)
    nif_prd_bp_4ch = nib.Nifti1Image(vol_4ch.astype(np.uint8), affine=affine_4ch)
    return nif_prd_bp_2ch, nif_prd_bp_4ch

def load_la_model(model_file: str, device_str: str = 'cpu') -> nn.Module:
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

    unet = ReconstructLaUNet3d(in_channel=1, out_channel=6)

    checkpoint = torch.load(model_file, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device, dtype=torch.float)
    if not is_cpu:
        unet.cuda()
    else:
        unet.cpu()
    
    return unet



def mr_lax_inference(unet, img_file_2ch, img_file_4ch, device_str: str):

    data_in = image_load_nifti_ori(img_file_2ch, img_file_4ch)
    affine = data_in.affine
    vol_in = np.flip(data_in.get_fdata(), axis=1)
    vol_in_la = np.zeros(np.shape(vol_in))
    vol_in_la[np.where(vol_in == 1)] = 1
    vol_in_lv = np.zeros(np.shape(vol_in))
    vol_in_lv[np.where(vol_in == 2)] = 5
    vol_in_ = vol_in_la + vol_in_lv
    test_x = np.zeros((1, 1, 128, 128, 128))
    test_x[0, 0, ...] = vol_in_ * 50

    # Move to appropriate device
    if device_str == 'cuda':
        t_x = Variable(torch.from_numpy(test_x).float().cuda())
    else:
        t_x = Variable(torch.from_numpy(test_x).float().cpu())

    output = unet(t_x)
    pred = output.detach().cpu().numpy()
    prd_tyi = np.flip(np.argmax(pred, axis=1).squeeze(), axis=1)
    nifti_la = nib.Nifti1Image(prd_tyi.astype(np.uint8), affine=affine)
    nif_prd_bp_2ch, nif_prd_bp_4ch = image_save_nifti_ori(img_file_2ch, img_file_4ch, prd_tyi)
    return data_in, nifti_la, nif_prd_bp_2ch, nif_prd_bp_4ch

def run_la_reconstruction(
    model,
    ch2_file: str,
    ch4_file: str,
    output_dir: str,
    output_prefix: str,
    device_str: str = 'cpu',
    compute_bp: bool = True
) -> dict:
    """
    Runs the LA-specific 3D reconstruction from 2CH and 4CH views.

    This function is a wrapper around the original `mr_lax_inference` logic,
    adapting it to our single-case, flexible-output pipeline.
    """
    
    outputs = OutputManager(output_dir=output_dir, output_prefix=output_prefix)
    
    # 2. Set the model to evaluation mode
    model.eval()
    
    # --- Execute the original core logic ---
    # We call the legacy function, but pass our pre-loaded model to it.
    nif_sp, nif_prd, nif_prd_bp_2ch, nif_prd_bp_4ch = mr_lax_inference(
        model, ch2_file, ch4_file, device_str
    )

    # --- Save the outputs using our manager ---
    # These keys must be added to the OutputManager config
    path_la_prediction = outputs.get_path('la_prediction')
    path_la_sparse_volume = outputs.get_path('la_sparse_volume')
    path_la_ch2_bp = outputs.get_path('la_ch2_bp')
    path_la_ch4_bp = outputs.get_path('la_ch4_bp')
    
    nib.save(nif_prd, path_la_prediction)
    nib.save(nif_sp, path_la_sparse_volume)
    
    if compute_bp:
        nib.save(nif_prd_bp_2ch, path_la_ch2_bp)
        nib.save(nif_prd_bp_4ch, path_la_ch4_bp)

    logger.info("LA Reconstruction complete!")
    return {
        'la_prediction': path_la_prediction,
        'la_sparse_volume': path_la_sparse_volume,
        'la_ch2_bp': path_la_ch2_bp if compute_bp else '',
        'la_ch4_bp': path_la_ch4_bp if compute_bp else ''
    }
