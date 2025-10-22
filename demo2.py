import sys
sys.path.append('core')
import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import h5py

def resize_image(img_chw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_chw: C x H x W numpy array    
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)
    resized_chw = np.transpose(resized_hwc, (2, 0, 1))
    
    return resized_chw

def resize_batch(batch_nchw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    return np.stack([resize_image(img, target_h, target_w, interpolation) for img in batch_nchw])

def read_h5_chunk(h5_file, dataset_name, start_idx, chunk_size=5):
    """
    Read a chunk of data from an HDF5 file.
    
    Args:
        h5_file (str): Path to HDF5 file
        dataset_name (str): Name of the dataset to read
        start_idx (int): Starting index to read from
        chunk_size (int): Maximum number of images to read (default: 5)
    
    Returns:
        tuple: (data chunk, actual_size)
            - data chunk: numpy array in NxCxHxW format
            - actual_size: number of images actually read (may be less than chunk_size at end of file)
    """
    with h5py.File(h5_file, 'r') as f:
        dataset = f[dataset_name]
        try:
            # Get the actual number of images we can read
            total_images = dataset.shape[0]
            actual_size = min(chunk_size, total_images - start_idx)
            
            if actual_size <= 0:
                return None, 0
                
            # Read the chunk
            chunk = dataset[start_idx:start_idx + actual_size]
            return chunk, actual_size
            
        except IndexError:
            return None, 0

def write_h5_chunk(h5_file, dataset_name, data, start_idx, shape=None, dtype=np.float32):
    """
    Write a chunk of data to an HDF5 file with gzip compression.
    
    Args:
        h5_file (str): Path to HDF5 file
        dataset_name (str): Name of the dataset to write
        data (numpy.ndarray): Data chunk in NxCxHxW format
        start_idx (int): Starting index to write at
        shape (tuple): Total shape of the dataset (N,C,H,W). Required only when creating new dataset
        dtype: Data type for the dataset (default: np.float32)
    """
    with h5py.File(h5_file, 'a') as f:
        if dataset_name not in f:
            if shape is None:
                raise ValueError("shape must be provided when creating a new dataset")
            # Create dataset with gzip compression if it doesn't exist
            f.create_dataset(dataset_name, 
                           shape=shape,
                           dtype=dtype,
                           compression='gzip',
                           compression_opts=4,
                           chunks=True)  # Let h5py choose optimal chunk size
        
        # Write the chunk
        dataset = f[dataset_name]
        end_idx = start_idx + len(data)
        dataset[start_idx:end_idx] = data

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
    
    stereo_params = np.load(args.stereo_params_npz_file, allow_pickle=True)
    P1 = stereo_params['P1']
    #P1[:2] *= args.scale
    f_left = P1[0,0]
    baseline = stereo_params['baseline']

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # left_images = np.load(args.left_imgs) #sorted(glob.glob(args.left_imgs, recursive=True))
    # right_images = np.load(args.right_imgs) #sorted(glob.glob(args.right_imgs, recursive=True))
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    prev_start_idx = 0
    if args.left_h5_file and args.right_h5_file:
        start_idx = 0
        chunk_size = 5
        while True:
            prev_start_idx = start_idx
            left_chunk, actual_left_size = read_h5_chunk(args.left_h5_file, 'rectified_lefts', start_idx, chunk_size)
            right_chunk, actual_right_size = read_h5_chunk(args.right_h5_file, 'rectified_rights', start_idx, chunk_size)
            assert actual_left_size == actual_right_size, f"left and right HDF5 chunks have different sizes: {actual_left_size} vs {actual_right_size}"
            if actual_left_size == 0:
                break
            start_idx += actual_left_size
            # try:
            #     with h5py.File(args.left_h5_file, 'r') as f:
            #         left_all = f['data'][()]   # or np.array(f['left'])
            #     with h5py.File(args.right_h5_file, 'r') as f:
            #         right_all = f['data'][()]
            # except Exception as e:            
            #     with h5py.File(args.left_h5_file, 'r') as f:
            #         left_all = f['left'][()]   # or np.array(f['left'])
            #     with h5py.File(args.right_h5_file, 'r') as f:
            #         right_all = f['right'][()]      
            print(left_chunk.shape, right_chunk.shape)
    
            if left_chunk.ndim==3:
                left_chunk = left_chunk[None]
                right_chunk = right_chunk[None]
            
            N,C,H,W = left_chunk.shape
            if args.process_only:
                N_stop = args.process_only
            else:
                N_stop = N
            N_max = N_stop
            # aspect ratio for Canon EOS 6D is 3/2. 3648
            # image size of about 1586x2379 works with batch_size of 1, 
            # with resize_factor of 2.3 at 28s/image, up to ~25 images.
            small_dim = min(H,W)
            large_dim = max(H,W)
            resize_factor = 2.5 # max(round(small_dim/1586,1), round(large_dim/2379,1))
            # resize_factor = 1.5
            print(f"Found {N} images in this chunk,  applying resize_factor {resize_factor} Saving files to {out_dir}.")
            
            disp_chunk = []
            depth_chunk = []

            with torch.no_grad():       
                #op_list = []
                #for i in range(0, N, args.batch_size): #tqdm(range(left_images.shape[0])):
                for i in tqdm(range(0, N, args.batch_size), desc="Processing batches"):  
                    img0 = left_chunk[i:i+args.batch_size]
                    img1 = right_chunk[i:i+args.batch_size]

                    if len(img0.shape)==3:
                        img0 = img0[None,...]

                    if len(img1.shape)==3:
                        img1 = img1[None,...]

                    img0 = resize_batch(img0, round(H/resize_factor) ,round(W/resize_factor))
                    img1 = resize_batch(img1, round(H/resize_factor), round(W/resize_factor))


                    img0 = torch.as_tensor(img0).cuda().float()
                    img1 = torch.as_tensor(img1).cuda().float()


                    # imfile1 = left_all[i].squeeze()
                    # imfile2 = right_all[i].squeeze()
                    # image1 = load_imarr(imfile1)
                    # image2 = load_imarr(imfile2)

                    padder = InputPadder(img0.shape, divis_by=32)
                    img0, img1 = padder.pad(img0, img1)   
                    print("Padded input shape:", img0.shape, img1.shape)         
                    disp_pr = model(img0, img1, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
                    disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()
                    depth = f_left*baseline/(disp_pr+1e-6)
                    # op_list.append(disp_pr)
                    # file_stem = "defom_{i}"#imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
                    # if args.save_numpy:
                    #     np.save(output_directory / f"{file_stem}.npy", disp_pr)
                    #plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')

                    disp_chunk.append(disp_pr)
                    depth_chunk.append(depth)
                    if i+args.batch_size >= N_stop:
                        N_max = i + img0.shape[0]
                        break
                # if args.save_numpy:
                # np.save(output_directory / f"defom_depth.npy", np.array(op_list))

            disp_chunk = np.concatenate(disp_chunk, axis=0).reshape(N_max,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)
            depth_chunk = np.concatenate(depth_chunk, axis=0).reshape(N_max,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)

            # with h5py.File(f'{args.out_dir}/leftview_disp_depth.h5', 'w') as f:
            #     f.create_dataset('disp', data=disp_chunk, compression='gzip')
            #     f.create_dataset('depth', data=depth_chunk, compression='gzip')
            write_h5_chunk(f'{args.out_dir}/leftview_disp_depth.h5', 'disp', disp_chunk, prev_start_idx, shape=(N_max,round(H/resize_factor),round(W/resize_factor)),dtype=np.float16)
            write_h5_chunk(f'{args.out_dir}/leftview_disp_depth.h5', 'depth', depth_chunk, prev_start_idx, shape=(N_max,round(H/resize_factor),round(W/resize_factor)),dtype=np.float16)

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
    parser.add_argument("--process_only",default=None,type=int)
    
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
