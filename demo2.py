import sys
sys.path.append('core')
import os
import argparse
import glob
import numpy as np
import h5py
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'

def load_imarr(imarr):
    #img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(imarr).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    stereo_params = np.load(args.stereo_params_npz_file, allow_pickle=True)
    P1 = stereo_params['P1']
    #P1[:2] *= args.scale
    f_left = P1[0,0]
    baseline = stereo_params['baseline']

    out_dir = Path(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # left_images = np.load(args.left_imgs) #sorted(glob.glob(args.left_imgs, recursive=True))
    # right_images = np.load(args.right_imgs) #sorted(glob.glob(args.right_imgs, recursive=True))
    
    
    if args.left_h5_file and args.right_h5_file:
      with h5py.File(args.left_h5_file, 'r') as f:
        left_all = f['left'][()]   # or np.array(f['left'])
      with h5py.File(args.right_h5_file, 'r') as f:
        right_all = f['right'][()]
      print(left_all.shape, right_all.shape)
    
    if left_all.ndim==3:
      left_all = left_all[None]
      right_all = right_all[None]
    
    N,H,W,C = left_all.shape
    print(f"Found {N} images. Saving files to {out_dir}.")
    disp_all = []
    depth_all = []

    with torch.no_grad():       
        op_list = []
        for i in range(0, N, args.batch_size): #tqdm(range(left_images.shape[0])):
            img0 = left_all[i:i+args.batch_size]
            img1 = right_all[i:i+args.batch_size]
            img0 = torch.as_tensor(img0).cuda().float().permute(0,3,1,2)
            img1 = torch.as_tensor(img1).cuda().float().permute(0,3,1,2)


            # imfile1 = left_all[i].squeeze()
            # imfile2 = right_all[i].squeeze()
            # image1 = load_imarr(imfile1)
            # image2 = load_imarr(imfile2)

            padder = InputPadder(img0.shape, divis_by=32)
            img0, img1 = padder.pad(img0, img1)

            with torch.no_grad():
                disp_pr = model(img0, img1, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()
            depth = f_left*baseline/(disp_pr+1e-6)
            # op_list.append(disp_pr)
            # file_stem = "defom_{i}"#imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", disp_pr)
            #plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')

            disp_all.append(disp_pr)
            depth_all.append(depth)
        
        # if args.save_numpy:
        # np.save(output_directory / f"defom_depth.npy", np.array(op_list))

    disp_all = np.concatenate(disp_all, axis=0).reshape(N,H,W).astype(np.float16)
    depth_all = np.concatenate(depth_all, axis=0).reshape(N,H,W).astype(np.float16)

    with h5py.File(f'{args.out_dir}/leftview_disp_depth.h5', 'w') as f:
      f.create_dataset('disp', data=disp_all, compression='gzip')
      f.create_dataset('depth', data=depth_all, compression='gzip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=5, type=int)    
    parser.add_argument('--left_h5_file', default="", type=str)
    parser.add_argument('--right_h5_file', default="", type=str)
    parser.add_argument('--stereo_params_npz_file', default = "", type = str)    
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--out_dir', default=f'../output/', type=str, help='the directory to save results')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays', default=True)
    #parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="demo/*_left.png")
    #parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="demo/*_right.png")
    #parser.add_argument('--output_directory', help="directory to save output", default="demo")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")

    # Architecture choices
    parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)


'''
import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'

def load_imarr(imarr):
    #img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(imarr).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    left_images = np.load(args.left_imgs) #sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = np.load(args.right_imgs) #sorted(glob.glob(args.right_imgs, recursive=True))
    print(f"Found {left_images.shape} images. Saving files to {output_directory}/")
    
    with torch.no_grad():       

        for i in tqdm(range(left_images.shape[0])):

            imfile1 = left_images[i].squeeze()
            imfile2 = right_images[i].squeeze()
            image1 = load_imarr(imfile1)
            image2 = load_imarr(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                disp_pr = model(image1, image2, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()

            file_stem = imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp_pr)
            plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="demo/*_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="demo/*_right.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")

    # Architecture choices
    parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
'''
