import os 
import random
from scipy.ndimage.morphology import binary_erosion
import numpy as np
import nibabel as nib

## LEGACY functions 
# data processing - PLANES
def lab_sax_plane(lab_sax):
    ns = np.shape(lab_sax)[-1]
    lab_sax_cs = []
    lab_sax_lb = []
    for ks in range(ns):
        lab_sax_ = lab_sax[..., ks]
        labs = np.unique(lab_sax_)
        # labs_ = [x for x in labs if x > 0]
        lab_sax_lb.append(labs)
        lab_sax_c = []
        for kl in labs:
            bm_i = lab_sax_ == kl
            # bm_i_ = np.subtract(bm_i, binary_erosion(bm_i).astype(int))
            pc_i = np.array(np.where(bm_i[..., np.newaxis]))
            pc_i[-1] = ks
            lab_sax_c.append(pc_i)
        lab_sax_cs.append(lab_sax_c)
    return lab_sax_cs, lab_sax_lb

def lab_lax_plane(lab_lax):
    lab_lax_ = lab_lax[..., 0]
    labs = np.unique(lab_lax_)
    # labs_ = [x for x in labs if x > 0]
    lab_lax_c = []
    for kl in labs:
        bm_i = lab_lax_ == kl
        # bm_i_ = np.subtract(bm_i, binary_erosion(bm_i).astype(int))
        pc_i = np.where(bm_i[..., np.newaxis])
        lab_lax_c.append(pc_i)
    return lab_lax_c

def load_lax_ps(ch2_file):
    data_lax = nib.load(ch2_file)
    lab_lax = data_lax.get_fdata()
    affine_lax = data_lax.affine
    lab_lax_c = lab_lax_plane(lab_lax)
    lax_pc = []
    for kc in lab_lax_c:
        ijk = np.array(kc)
        ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
        xyz1 = np.matmul(affine_lax, ijk1)
        xyz = xyz1[0:3, ...]
        lax_pc.append(xyz)
    # ipp, ipo
    ipp = np.matmul(affine_lax, np.array([[0], [0], [0], [1]]))[0:3]
    ip0 = np.matmul(affine_lax, np.array([[10], [0], [0], [1]]))[0:3]
    ip1 = np.matmul(affine_lax, np.array([[0], [10], [0], [1]]))[0:3]
    v0 = (ip0 - ipp) / np.linalg.norm(ip0 - ipp)
    v1 = (ip1 - ipp) / np.linalg.norm(ip1 - ipp)
    ipo = np.concatenate((v0, v1))
    pxs = [np.linalg.norm(ip0 - ipp) / 10, np.linalg.norm(ip1 - ipp) / 10]
    return lax_pc, ipp, ipo, pxs, lab_lax

def load_sax_ps(sax_file):
    data_sax = nib.load(sax_file)
    lab_sax = data_sax.get_fdata()
    affine_sax = data_sax.affine
    lab_sax_cs, lab_sax_lb = lab_sax_plane(lab_sax)
    sax_pcs = []
    for lab_sax_c in lab_sax_cs:
        sax_pc = []
        for kc in lab_sax_c:
            ijk = np.array(kc)
            ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
            xyz1 = np.matmul(affine_sax, ijk1)
            xyz = xyz1[0:3, ...]
            sax_pc.append(xyz)
        sax_pcs.append(sax_pc)
    # ipp, ipo
    ipp = []
    for ks in range(np.shape(lab_sax)[-1]):
        ipp.append(np.matmul(affine_sax, np.array([[0], [0], [ks], [1]]))[0:3])
    ip0 = np.matmul(affine_sax, np.array([[10], [0], [0], [1]]))[0:3]
    ip1 = np.matmul(affine_sax, np.array([[0], [10], [0], [1]]))[0:3]
    v0 = (ip0 - ipp[0]) / np.linalg.norm(ip0 - ipp[0])
    v1 = (ip1 - ipp[0]) / np.linalg.norm(ip1 - ipp[0])
    ipo = np.concatenate((v0, v1))
    pxs = [np.linalg.norm(ip0 - ipp[0]) / 10, np.linalg.norm(ip1 - ipp[0]) / 10]
    return sax_pcs, lab_sax_lb, ipp, ipo, pxs, lab_sax

# data processing - CONTOURS
def lab_sax_contour(lab_sax):
    ns = np.shape(lab_sax)[-1]
    lab_sax_cs = []
    lab_sax_lb = []
    for ks in range(ns):
        lab_sax_ = lab_sax[..., ks]
        labs = np.unique(lab_sax_)
        labs_ = [x for x in labs if x > 0]
        lab_sax_lb.append(labs_)
        lab_sax_c = []
        for kl in labs_:
            bm_i = lab_sax_ == kl
            bm_i_ = np.subtract(bm_i, binary_erosion(bm_i).astype(int))
            pc_i = np.array(np.where(bm_i_[..., np.newaxis]))
            pc_i[-1] = ks
            lab_sax_c.append(pc_i)
        lab_sax_cs.append(lab_sax_c)
    return lab_sax_cs, lab_sax_lb

def lab_lax_contour(lab_lax):
    lab_lax_ = lab_lax[..., 0]
    labs = np.unique(lab_lax_)
    labs_ = [x for x in labs if x > 0]
    lab_lax_c = []
    for kl in labs_:
        bm_i = lab_lax_ == kl
        bm_i_ = np.subtract(bm_i, binary_erosion(bm_i).astype(int))
        pc_i = np.where(bm_i_[..., np.newaxis])
        lab_lax_c.append(pc_i)
    return lab_lax_c

def load_lax_pc(ch2_file):
    data_lax = nib.load(ch2_file)
    lab_lax = data_lax.get_fdata()
    affine_lax = data_lax.affine
    lab_lax_c = lab_lax_contour(lab_lax)
    lax_pc = []
    for kc in lab_lax_c:
        ijk = np.array(kc)
        ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
        xyz1 = np.matmul(affine_lax, ijk1)
        xyz = xyz1[0:3, ...]
        lax_pc.append(xyz)
    return lax_pc, affine_lax

def load_sax_pc(sax_file):
    data_sax = nib.load(sax_file)
    lab_sax = data_sax.get_fdata()
    affine_sax = data_sax.affine
    lab_sax_cs, lab_sax_lb = lab_sax_contour(lab_sax)
    sax_pcs = []
    for lab_sax_c in lab_sax_cs:
        sax_pc = []
        for kc in lab_sax_c:
            ijk = np.array(kc)
            ijk1 = np.concatenate((ijk, np.ones((1, np.shape(ijk)[1]))), axis=0)
            xyz1 = np.matmul(affine_sax, ijk1)
            xyz = xyz1[0:3, ...]
            sax_pc.append(xyz)
        sax_pcs.append(sax_pc)
    return sax_pcs, lab_sax_lb, affine_sax

def rand_tri(ch2_ps):
    i0 = random.choice(list(range(len(ch2_ps[0][0]))))
    x0 = np.transpose(ch2_ps[0])[i0]
    d0 = np.linalg.norm(np.transpose(ch2_ps[0]) - x0, axis=1)
    i1 = np.argmax(d0)
    x1 = np.transpose(ch2_ps[0])[i1]
    d1 = np.linalg.norm(np.transpose(ch2_ps[0]) - x1, axis=1)
    i2 = np.argmax(d0 + d1)
    x2 = np.transpose(ch2_ps[0])[i2]
    n = np.cross(x2 - x0, x1 - x0) / np.linalg.norm(np.cross(x2 - x0, x1 - x0))
    return n


def vol_grid_gen(ch2_ps, ch4_ps, sax_ps,
                 ch2_pc, ch4_pc, sax_pc,
                 sax_ipp, sax_ipo, sax_pxs, sax_lab,
                 ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                 ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab):
    # ax
    n_2ch = rand_tri(ch2_ps)
    n_4ch = rand_tri(ch4_ps)
    ax_l0 = np.cross(n_2ch, n_4ch) / np.linalg.norm(np.cross(n_2ch, n_4ch))
    lv_2ch = np.mean(ch2_ps[1], axis=1)
    la_2ch = np.mean(ch2_ps[3], axis=1)
    if np.dot(la_2ch - lv_2ch, ax_l0) > 0:
        ax_ab = ax_l0
    else:
        ax_ab = - ax_l0
    lv_4ch = np.mean(ch4_ps[1], axis=1)
    rv_4ch = np.mean(ch4_ps[3], axis=1)
    ax_r0 = np.cross(n_4ch, ax_ab) / np.linalg.norm(np.cross(n_4ch, ax_ab))
    if np.dot(rv_4ch - lv_4ch, ax_r0) > 0:
        ax_lr = ax_r0
    else:
        ax_lr = - ax_r0
    ax_fb = np.cross(ax_ab, ax_lr) / np.linalg.norm(np.cross(ax_ab, ax_lr))
    # og
    cs = []
    for k2 in ch2_pc:
        for kl in range(np.shape(k2)[1]):
            cs.append(np.array(k2)[:, kl])
    for k4 in ch4_pc:
        for kl in range(np.shape(k4)[1]):
            cs.append(np.array(k4)[:, kl])

    min_lax = np.min(cs, axis=0)
    max_lax = np.max(cs, axis=0)
    o_c = np.mean((min_lax, max_lax), axis=0)
    d_ab = np.dot(cs - o_c, ax_ab).max() - np.dot(cs - o_c, ax_ab).min()
    d_lr = np.dot(cs - o_c, ax_lr).max() - np.dot(cs - o_c, ax_lr).min()
    d_fb = np.dot(cs - o_c, ax_fb).max() - np.dot(cs - o_c, ax_fb).min()
    c_ab = (np.dot(cs - o_c, ax_ab).max() + np.dot(cs - o_c, ax_ab).min()) / 2
    c_lr = (np.dot(cs - o_c, ax_lr).max() + np.dot(cs - o_c, ax_lr).min()) / 2
    c_fb = (np.dot(cs - o_c, ax_fb).max() + np.dot(cs - o_c, ax_fb).min()) / 2
    o_c_ = o_c + c_ab * ax_ab + c_lr * ax_lr + c_fb * ax_fb
    vs = np.max([d_ab, d_lr, d_fb]) * 1.2 / 160   # 10% margin around
    # vol dense
    ii = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    jj = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    kk = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    iv, jv, kv = np.meshgrid(ii, jj, kk)
    ijk_v = np.array([np.resize(iv, np.size(iv)), np.resize(jv, np.size(jv)), np.resize(kv, np.size(kv))]).transpose()
    vm = np.array([ax_ab, ax_lr, ax_fb]).transpose()
    xyz_v = o_c_ + np.dot(vm, ijk_v.transpose()).transpose() * vs
    # vol sparse
    vol_sp = np.zeros((160, 160, 160))
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
    pq_2ch = np.transpose([np.dot(xyz_2ch - np.transpose(ch2_ipp), v0_s) / ch2_pxs[0],
                           np.dot(xyz_2ch - np.transpose(ch2_ipp), v1_s) / ch2_pxs[1]]).squeeze().round()
    px2vx_2ch = [0, 1, 2, 5]
    for ki in range(len(i_2ch)):
        try:
            vol_sp[ijk_v_[i_2ch[ki]][0].astype(int),
                   ijk_v_[i_2ch[ki]][1].astype(int),
                   ijk_v_[i_2ch[ki]][2].astype(int)] = px2vx_2ch[ch2_lab[pq_2ch[ki][0].astype(int),
                                                                         pq_2ch[ki][1].astype(int),
                                                                         0].astype(int)]
        except:
            print('Out of box voxels.', end='\r')
    # 4ch
    d4 = np.dot(xyz_v - ch4_pc[0][:, 0], n_4ch)
    i_4ch = np.where(np.abs(d4) <= vs)[0]
    xyz_4ch = xyz_v[i_4ch, :]
    v0_s = ch4_ipo[0:3]
    v1_s = ch4_ipo[3:]
    pq_4ch = np.transpose([np.dot(xyz_4ch - np.transpose(ch4_ipp), v0_s) / ch4_pxs[0],
                           np.dot(xyz_4ch - np.transpose(ch4_ipp), v1_s) / ch4_pxs[1]]).squeeze().round()
    px2vx_4ch = [0, 1, 2, 3, 5, 6]
    for ki in range(len(i_4ch)):
        try:
            vol_sp[ijk_v_[i_4ch[ki]][0].astype(int),
                   ijk_v_[i_4ch[ki]][1].astype(int),
                   ijk_v_[i_4ch[ki]][2].astype(int)] = px2vx_4ch[ch4_lab[pq_4ch[ki][0].astype(int),
                                                                         pq_4ch[ki][1].astype(int),
                                                                         0].astype(int)]
        except:
            print('Out of box voxels.', end='\r')
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
        pq_sax = np.transpose([np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v0_s) / sax_pxs[0],
                               np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v1_s) / sax_pxs[1]]).squeeze().round()
        px2vx_sax = [0, 1, 2, 3]
        for ki in range(len(i_ks)):
            try:
                if sax_lab[pq_sax[ki][0].astype(int), pq_sax[ki][1].astype(int), ks] > 0:
                    vol_sp[ijk_v_[i_ks[ki]][0].astype(int),
                           ijk_v_[i_ks[ki]][1].astype(int),
                           ijk_v_[i_ks[ki]][2].astype(int)] = px2vx_sax[sax_lab[pq_sax[ki][0].astype(int),
                                                                                pq_sax[ki][1].astype(int),
                                                                                ks].astype(int)]
            except:
                print('Out of box voxels.', end='\r')
    return vol_sp, affine_3d


def vol_grid_bp(ch2_ps, ch4_ps, sax_ps,
                ch2_pc, ch4_pc, sax_pc,
                sax_ipp, sax_ipo, sax_pxs, sax_lab,
                ch2_ipp, ch2_ipo, ch2_pxs, ch2_lab,
                ch4_ipp, ch4_ipo, ch4_pxs, ch4_lab,
                vol_pr):
    # ax
    n_2ch = rand_tri(ch2_ps)
    n_4ch = rand_tri(ch4_ps)
    ax_l0 = np.cross(n_2ch, n_4ch) / np.linalg.norm(np.cross(n_2ch, n_4ch))
    lv_2ch = np.mean(ch2_ps[1], axis=1)
    la_2ch = np.mean(ch2_ps[3], axis=1)
    if np.dot(la_2ch - lv_2ch, ax_l0) > 0:
        ax_ab = ax_l0
    else:
        ax_ab = - ax_l0
    lv_4ch = np.mean(ch4_ps[1], axis=1)
    rv_4ch = np.mean(ch4_ps[3], axis=1)
    ax_r0 = np.cross(n_4ch, ax_ab) / np.linalg.norm(np.cross(n_4ch, ax_ab))
    if np.dot(rv_4ch - lv_4ch, ax_r0) > 0:
        ax_lr = ax_r0
    else:
        ax_lr = - ax_r0
    ax_fb = np.cross(ax_ab, ax_lr) / np.linalg.norm(np.cross(ax_ab, ax_lr))
    # og
    cs = []
    for k2 in ch2_pc:
        for kl in range(np.shape(k2)[1]):
            cs.append(np.array(k2)[:, kl])
    for k4 in ch4_pc:
        for kl in range(np.shape(k4)[1]):
            cs.append(np.array(k4)[:, kl])

    min_lax = np.min(cs, axis=0)
    max_lax = np.max(cs, axis=0)
    o_c = np.mean((min_lax, max_lax), axis=0)
    d_ab = np.dot(cs - o_c, ax_ab).max() - np.dot(cs - o_c, ax_ab).min()
    d_lr = np.dot(cs - o_c, ax_lr).max() - np.dot(cs - o_c, ax_lr).min()
    d_fb = np.dot(cs - o_c, ax_fb).max() - np.dot(cs - o_c, ax_fb).min()
    c_ab = (np.dot(cs - o_c, ax_ab).max() + np.dot(cs - o_c, ax_ab).min()) / 2
    c_lr = (np.dot(cs - o_c, ax_lr).max() + np.dot(cs - o_c, ax_lr).min()) / 2
    c_fb = (np.dot(cs - o_c, ax_fb).max() + np.dot(cs - o_c, ax_fb).min()) / 2
    o_c_ = o_c + c_ab * ax_ab + c_lr * ax_lr + c_fb * ax_fb
    vs = np.max([d_ab, d_lr, d_fb]) * 1.2 / 160   # 10% margin around
    # vol dense
    ii = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    jj = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    kk = np.linspace(0, int(160 - 1), 160) - int(160 / 2)
    iv, jv, kv = np.meshgrid(ii, jj, kk)
    ijk_v = np.array([np.resize(iv, np.size(iv)), np.resize(jv, np.size(jv)), np.resize(kv, np.size(kv))]).transpose()
    vm = np.array([ax_ab, ax_lr, ax_fb]).transpose()
    xyz_v = o_c_ + np.dot(vm, ijk_v.transpose()).transpose() * vs
    # vol dense
    vol_ds = vol_pr
    ijk_v_ = ijk_v + 80
    # 2ch
    d2 = np.dot(xyz_v - ch2_pc[0][:, 0], n_2ch)
    i_2ch = np.where(np.abs(d2) <= vs)[0]
    xyz_2ch = xyz_v[i_2ch, :]
    v0_s = ch2_ipo[0:3]
    v1_s = ch2_ipo[3:]
    pq_2ch = np.transpose([np.dot(xyz_2ch - np.transpose(ch2_ipp), v0_s) / ch2_pxs[0],
                           np.dot(xyz_2ch - np.transpose(ch2_ipp), v1_s) / ch2_pxs[1]]).squeeze().round()
    ch2_bp = np.zeros(np.shape(ch2_lab))
    px2vx_2ch = [0, 1, 2, 0, 0, 3, 0, 0, 0]
    for ki in range(len(i_2ch)):
        try:
            ch2_bp[pq_2ch[ki][0].astype(int),
                   pq_2ch[ki][1].astype(int),
                   0] = px2vx_2ch[vol_ds[ijk_v_[i_2ch[ki]][0].astype(int),
                                         ijk_v_[i_2ch[ki]][1].astype(int),
                                         ijk_v_[i_2ch[ki]][2].astype(int)].astype(int)]
        except:
            print('Out of box voxels.', end='\r')
    # 4ch
    d4 = np.dot(xyz_v - ch4_pc[0][:, 0], n_4ch)
    i_4ch = np.where(np.abs(d4) <= vs)[0]
    xyz_4ch = xyz_v[i_4ch, :]
    v0_s = ch4_ipo[0:3]
    v1_s = ch4_ipo[3:]
    pq_4ch = np.transpose([np.dot(xyz_4ch - np.transpose(ch4_ipp), v0_s) / ch4_pxs[0],
                           np.dot(xyz_4ch - np.transpose(ch4_ipp), v1_s) / ch4_pxs[1]]).squeeze().round()
    ch4_bp = np.zeros(np.shape(ch2_lab))
    px2vx_4ch = [0, 1, 2, 3, 0, 4, 5, 0, 0]
    for ki in range(len(i_4ch)):
        try:
            ch4_bp[pq_4ch[ki][0].astype(int),
                   pq_4ch[ki][1].astype(int),
                   0] = px2vx_4ch[vol_ds[ijk_v_[i_4ch[ki]][0].astype(int),
                                         ijk_v_[i_4ch[ki]][1].astype(int),
                                         ijk_v_[i_4ch[ki]][2].astype(int)].astype(int)]
        except:
            print('Out of box voxels.', end='\r')
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
        pq_sax = np.transpose([np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v0_s) / sax_pxs[0],
                               np.dot(xyz_ks - np.transpose(sax_ipp[ks]), v1_s) / sax_pxs[1]]).squeeze().round()
        px2vx_sax = [0, 1, 2, 3, 0, 0, 0, 0, 0]
        for ki in range(len(i_ks)):
            try:
                sax_bp[pq_sax[ki][0].astype(int),
                       pq_sax[ki][1].astype(int),
                       ks] = px2vx_sax[vol_ds[ijk_v_[i_ks[ki]][0].astype(int),
                                              ijk_v_[i_ks[ki]][1].astype(int),
                                              ijk_v_[i_ks[ki]][2].astype(int)].astype(int)]
            except:
                print('Out of box voxels.', end='\r')
    return ch2_bp, ch4_bp, sax_bp