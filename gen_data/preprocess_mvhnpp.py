import os
from pathlib import Path
import json
import numpy as np
import smplx
import torch

def conv_mvhumannetpp_cameras_to_thuman(mvhn_subj_path):
    cameras = sorted(os.listdir(mvhn_subj_path / 'cameras'))
    calibrations = {}
    for cam in cameras:
        calibration = np.load(mvhumannet_root / subj / 'cameras' / cam / 'camera.npz')
        K = np.array(calibration['intrinsic']).flatten().tolist()
        ext = calibration['extrinsic']
        R = np.array(ext[:3, :3]).flatten().tolist()
        T = np.array(ext[:3, 3]).tolist()
        cam_cal = {'K': K, 'R': R, 'T': T,
               'distCoeff': [0.0, 0.0, 0.0, 0.0],
               'imgSize': [1024, 1024],
               'rectifyAlpha': 0.0}
        calibrations[cam] = cam_cal
    with open(mvhn_subj_path / 'calibrations.json', 'w') as f:
        json.dump(calibrations, f)


def convert_hand_pca_to_rotation(pca_params, side='right'):
    """
    Converts 6-dim PCA parameters to 45-dim Axis-Angle parameters.
    
    Args:
        pca_params (numpy array): Shape (1, 6) or (N, 6)
        side (str): 'left' or 'right'
    
    Returns:
        numpy array: Shape (1, 45)
    """
    # --- CONFIGURATION ---
    MODEL_FOLDER = '../smpl_files' # Path to folder containing SMPLX_NEUTRAL.npz
    GENDER = 'neutral'
    
    model = smplx.create(
        MODEL_FOLDER, 
        model_type='smplx', 
        gender=GENDER, 
        use_pca=True,
        num_pca_comps=6
    )

    # Convert input to tensor
    pca_tensor = torch.tensor(pca_params, dtype=torch.float32)
    
    # 2. Extract the specific data for the requested hand
    if side == 'right':
        mean_pose = model.right_hand_mean 
        components = model.right_hand_components 
    else:
        mean_pose = model.left_hand_mean
        components = model.left_hand_components

    active_components = components[:6, :]
    delta = torch.matmul(pca_tensor, active_components)
    full_pose = delta + mean_pose
    
    return full_pose.detach().numpy()


def conv_mvhumannetpp_smplx_to_thuman(mvhn_subj_path):
    frames = sorted(os.listdir(mvhn_subj_path / 'smplx_params_new'))
    smplx_all_frames = [np.load(mvhn_subj_path / 'smplx_params_new' / fr) for fr in frames]
    betas           = smplx_all_frames[0]['betas']
    transl          = np.concatenate([smplx_fr['transl'] for smplx_fr in smplx_all_frames])
    global_orient   = np.concatenate([smplx_fr['global_orient'] for smplx_fr in smplx_all_frames])
    body_pose       = np.concatenate([smplx_fr['body_pose'] for smplx_fr in smplx_all_frames])
    jaw_pose        = np.concatenate([smplx_fr['jaw_pose'] for smplx_fr in smplx_all_frames])
    expression      = np.concatenate([smplx_fr['expression'] for smplx_fr in smplx_all_frames])
    left_hand_pose  = np.concatenate([convert_hand_pca_to_rotation(smplx_fr['left_hand_pose'], 'left')
                                      for smplx_fr in smplx_all_frames])
    right_hand_pose = np.concatenate([convert_hand_pca_to_rotation(smplx_fr['right_hand_pose'], 'right')
                                      for smplx_fr in smplx_all_frames])
    
    np.savez(mvhn_subj_path / 'smpl_params.npz',
             betas=betas,
             global_orient=global_orient,
             transl=transl,
             body_pose=body_pose,
             jaw_pose=jaw_pose,
             expression=expression,
             left_hand_pose=left_hand_pose,
             right_hand_pose=right_hand_pose)


if __name__ == '__main__':
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mvhn_subj_path', type = str, help = 'Path to a single MVHumanNet++ subject.')
    arg_parser.add_argument('--mvhn_root', type = str, help = 'Path to folder containing multiple subjects.') 
    args = arg_parser.parse_args()

    if args.mvhn_subj_path is not None:
        mvhn_subj_paths = [Path(args.mvhn_subj_path)]
    else:
        mvhn_root = Path(args.mvhn_root)
        subjects = sorted(os.listdir(mvhn_root))
        mvhn_subj_paths = [mvhn_root / subj for subj in subjects]
        
    for mvhn_subj_path in mvhn_subj_paths:
        conv_mvhumannetpp_cameras_to_thuman(mvhn_subj_path)
        conv_mvhumannetpp_smplx_to_thuman(mvhn_subj_path)