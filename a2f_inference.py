import os
import cv2
import yaml
import shutil
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn.functional as F

from modules.a2f_net import A2FDiffNet
from config.a2f_diffusion_config import get_cfg_defaults
from utils.video import images2video, add_audio_to_video
from utils.util import add_suffix_name

from modules.spade_generator import SPADEDecoder
from modules.warping_network import WarpingNetwork
from modules.motion_extractor import MotionExtractor
from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from utils.audio_processor import AudioProcessor
from utils.camera import headpose_pred_to_degree, get_rotation_matrix
from utils.cropper import Cropper
from config.crop_config import CropConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lp_exp_idxs = [
    3, 4, 5, 6, 7, 8, 10, 13, 17, 18, 19, 20, 26, 28, 29, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
]

lp_motion_idxs = [
    3, 4, 5, 6, 7, 8, 10, 13, 17, 18, 19, 20, 26, 28, 29, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69
]


class PlaymatePipeline():
    def __init__(self, weights_dir=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'Device "{device}" requested but CUDA is not available; use device="cpu".')
        self.device = device
        self.weights_dir = weights_dir or os.path.join(BASE_DIR, 'pretrained_weights')
        self.n_motion = 50
        self.prev_n_motion = 5
        self.cfg_scale = [1.5, 1.5]

        checkpoint_paths = {
            'motion_extractor': os.path.join(self.weights_dir, 'base_models', 'motion_extractor.pth'),
            'appearance_feature_extractor': os.path.join(self.weights_dir, 'base_models', 'appearance_feature_extractor.pth'),
            'warping_module': os.path.join(self.weights_dir, 'base_models', 'warping_module.pth'),
            'spade_generator': os.path.join(self.weights_dir, 'base_models', 'spade_generator.pth'),
            'playmate': os.path.join(self.weights_dir, 'playmate', 'playmate.pt'),
        }
        required_paths = [
            os.path.join(BASE_DIR, 'config', 'models.yaml'),
            *checkpoint_paths.values(),
            os.path.join(self.weights_dir, 'wav2vec2-base-960h'),
            os.path.join(self.weights_dir, 'audio_separator', 'Kim_Vocal_2.onnx'),
            os.path.join(BASE_DIR, 'data', 'lip_array.pkl'),
            os.path.join(BASE_DIR, 'data', 'lp_motion_mean.pt'),
            os.path.join(BASE_DIR, 'data', 'lp_motion_std.pt'),
            os.path.join(BASE_DIR, 'data', 'lp_motion_norm_stand.pt'),
        ]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                'Missing required files:\n  - ' + '\n  - '.join(missing)
                + '\nDownload the pretrained weights first (see README).'
            )

        with open(os.path.join(BASE_DIR, 'config', 'models.yaml'), 'r') as f:
            model_config = yaml.safe_load(f)['model_params']
        self.motion_extractor = MotionExtractor(**model_config['motion_extractor_params']).to(self.device)
        self.appearance_feature_extractor = AppearanceFeatureExtractor(
            **model_config['appearance_feature_extractor_params']
        ).to(self.device)
        self.warping_module = WarpingNetwork(**model_config['warping_module_params']).to(self.device)
        self.spade_generator = SPADEDecoder(**model_config['spade_generator_params']).to(self.device)
        self.motion_extractor.load_state_dict(
            torch.load(checkpoint_paths['motion_extractor'], map_location=self.device, weights_only=True))
        self.appearance_feature_extractor.load_state_dict(
            torch.load(checkpoint_paths['appearance_feature_extractor'], map_location=self.device, weights_only=True))
        self.warping_module.load_state_dict(
            torch.load(checkpoint_paths['warping_module'], map_location=self.device, weights_only=True))
        self.spade_generator.load_state_dict(
            torch.load(checkpoint_paths['spade_generator'], map_location=self.device, weights_only=True))
        self.motion_extractor.eval()
        self.appearance_feature_extractor.eval()
        self.warping_module.eval()
        self.spade_generator.eval()

        self.crop_cfg = CropConfig()
        self._cropper = None  # lazily created on first use (crop_face=True)

        with open(os.path.join(BASE_DIR, 'data', 'lip_array.pkl'), 'rb') as f:
            self.lip_array = pkl.load(f)
            self.lip_array = self.lip_array.reshape(1, 1, -1)
            self.lip_array = torch.from_numpy(self.lip_array).to(self.device)

        self.audio_processor = AudioProcessor(
            os.path.join(self.weights_dir, 'wav2vec2-base-960h'),
            os.path.join(self.weights_dir, 'audio_separator'),
            'Kim_Vocal_2.onnx',
            device=self.device,
        )

        a2f_diff_cfg = get_cfg_defaults()
        self.a2f_model = A2FDiffNet(a2f_diff_cfg).to(self.device)
        self.a2f_model.load_state_dict(
            torch.load(checkpoint_paths['playmate'], map_location=self.device, weights_only=True)
        )
        self.a2f_model.eval()

        self.lp_motion_mean = torch.load(
            os.path.join(BASE_DIR, 'data', 'lp_motion_mean.pt'), weights_only=True
        ).reshape(1, 1, -1).to(self.device)  # (1, 1, 63)
        self.lp_motion_std = torch.load(
            os.path.join(BASE_DIR, 'data', 'lp_motion_std.pt'), weights_only=True
        ).reshape(1, 1, -1).to(self.device)  # (1, 1, 63)
        self.lp_motion_norm_stand = torch.load(
            os.path.join(BASE_DIR, 'data', 'lp_motion_norm_stand.pt'), weights_only=True
        ).reshape(1, 4, -1).to(self.device)
        self.lp_motion_mean = torch.cat([self.lp_motion_mean, self.lp_motion_norm_stand[:, 0:1, 63:]], dim=2)
        self.lp_motion_std = torch.cat([self.lp_motion_std, self.lp_motion_norm_stand[:, 1:2, 63:]], dim=2)

    @property
    def cropper(self):
        if self._cropper is None:
            self._cropper = Cropper(crop_cfg=self.crop_cfg)
        return self._cropper

    def img_tensor2numpy(self, imgs):
        imgs = np.transpose(imgs.cpu().numpy(), [0, 2, 3, 1])  # Bx3x512x512 -> Bx512x512x3
        imgs = np.clip(imgs, 0, 1)  # clip to 0~1
        imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return imgs

    def get_img_lp_motion(self, img):
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.
        imgs = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            f_s = self.appearance_feature_extractor(imgs)[0]
            kp_info = self.motion_extractor(imgs)
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # (B, 1)
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # (B, 1)
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # (B, 1)
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1)  # (B, N*3)
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1)  # (B, N*3)
            lp_motion = torch.cat(
                [kp_info['exp'], kp_info['pitch'], kp_info['yaw'], kp_info['roll'], kp_info['t'], kp_info['scale']],
                dim=1
            ).detach()[0]
        lp_canonical_kp = kp_info['kp'].detach()

        return f_s, lp_motion, lp_canonical_kp

    def normalize_standard(self, x):
        x = (x - self.lp_motion_mean) / self.lp_motion_std

        return x

    def reverse_normalize_standard(self, x):
        x = x * self.lp_motion_std + self.lp_motion_mean

        return x

    @torch.no_grad()
    def extract_driven_pose(self, video_path, data_num=None, return_dri=False):
        if video_path.endswith('.mp4'):
            video_cap = cv2.VideoCapture(video_path)
            if video_cap.get(cv2.CAP_PROP_FPS) != 25.0:
                raise ValueError('Driven video fps is not equal to 25 fps')

            frames = []
            while True:
                _, f = video_cap.read()
                if f is None:
                    break
                frames.append(f[:, :, ::-1])
                if data_num is not None and len(frames) >= data_num:
                    break

            data_num = len(frames)

            driven_poses = []
            batch_size = 8
            for idx in tqdm(range(0, data_num, batch_size)):
                imgs = []
                for img in frames[idx:idx + batch_size]:
                    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
                    img = torch.from_numpy(img).permute(2, 0, 1) / 255.
                    imgs.append(img)
                imgs = torch.stack(imgs).to(self.device)

                with torch.no_grad():
                    kp_info = self.motion_extractor(imgs)
                    bs = kp_info['kp'].shape[0]
                    kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # (B, 1)
                    kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # (B, 1)
                    kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # (B, 1)
                    driven_pose = torch.cat(
                        [kp_info['pitch'], kp_info['yaw'], kp_info['roll'], kp_info['t'], kp_info['scale']],
                        dim=1
                    ).detach().cpu()

                driven_poses.append(driven_pose)

            driven_poses = torch.cat(driven_poses)

            if return_dri:
                return driven_poses, frames
            else:
                return driven_poses

        else:
            img = cv2.imread(video_path)
            if img is None:
                raise FileNotFoundError(f'Failed to read pose-driven image: {video_path}')
            img = img[..., ::-1]
            img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
            img = torch.from_numpy(img).permute(2, 0, 1) / 255.
            imgs = img.unsqueeze(0).to(self.device)
            with torch.no_grad():
                kp_info = self.motion_extractor(imgs)
                bs = kp_info['kp'].shape[0]
                kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # (B, 1)
                kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # (B, 1)
                kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # (B, 1)
                driven_poses = torch.cat(
                    [kp_info['pitch'], kp_info['yaw'], kp_info['roll'], kp_info['t'], kp_info['scale']],
                    dim=1
                ).detach().cpu()

            return driven_poses

    def motion_modulation(self, model_preds, canonical_lp_motion, no_lipoffset, pose_driven_video_path,
                          rotate=False, fix_pose=False):
        if pose_driven_video_path is not None:
            driven_poses = self.extract_driven_pose(pose_driven_video_path).unsqueeze(1)
            driven_pose_len = len(driven_poses)
        else:
            driven_poses = None
            driven_pose_len = None

        if rotate and fix_pose:
            rotate_angle = self.lp_motion_mean.clone().repeat(len(model_preds), 1, 1)
            for idx, i in enumerate(range(-30, 30, 1)):
                rotate_angle[idx * 4:(idx + 1) * 4, :, 64:65] = i
            rotate_preds = model_preds.clone()
            rotate_preds[..., 63:] = rotate_preds[..., 63:] * self.lp_motion_std[..., 63:] + rotate_angle[..., 63:]

            rotate_angle = self.lp_motion_mean.clone()
            rotate_angle[..., 63:66] = 0
            fix_pose_preds = model_preds.clone()
            fix_pose_preds[..., 63:] = fix_pose_preds[..., 63:] * self.lp_motion_std[..., 63:] + rotate_angle[..., 63:]

        elif rotate and not fix_pose:
            rotate_angle = self.lp_motion_mean.clone().repeat(len(model_preds), 1, 1)
            for idx, i in enumerate(range(-30, 30, 1)):
                rotate_angle[idx * 4:(idx + 1) * 4, :, 64:65] = i
            model_preds[..., 63:] = model_preds[..., 63:] * self.lp_motion_std[..., 63:] + rotate_angle[..., 63:]
            rotate_preds = None
            fix_pose_preds = None
        elif fix_pose and not rotate:
            rotate_angle = self.lp_motion_mean.clone()
            rotate_angle[..., 63:66] = 0
            model_preds[..., 63:] = model_preds[..., 63:] * self.lp_motion_std[..., 63:] + rotate_angle[..., 63:]
            rotate_preds = None
            fix_pose_preds = None
        else:
            tmp_lp_motion_mean = canonical_lp_motion.clone().reshape(1, 1, -1)
            if driven_poses is not None and not pose_driven_video_path.endswith('.mp4'):
                tmp_lp_motion_mean[..., 63:-1] = driven_poses[..., :-1]
            model_preds[..., 63:] = \
                model_preds[..., 63:] * self.lp_motion_std[..., 63:] + tmp_lp_motion_mean[..., 63:]
            rotate_preds = None
            fix_pose_preds = None

        if not no_lipoffset:
            model_preds[..., lp_exp_idxs] -= self.lip_array[..., lp_exp_idxs] * 0.6

        if driven_poses is not None and pose_driven_video_path.endswith('.mp4'):
            model_preds[:driven_pose_len, :, 63:-1] = driven_poses[..., :-1]

        return model_preds, rotate_preds, fix_pose_preds

    def warp_and_generate(self, x_c, x_d, f_s):
        bs = 50
        T = x_d.shape[0]

        out_imgs = []
        for mb_idx in tqdm(range(0, T, bs)):
            mb_f_s = f_s[mb_idx:mb_idx + bs]
            mb_x_c = x_c[mb_idx:mb_idx + bs]
            mb_x_d = x_d[mb_idx:mb_idx + bs]
            ret_dct = self.warping_module(mb_f_s, kp_source=mb_x_c, kp_driving=mb_x_d)
            out_img = self.spade_generator(feature=ret_dct['out'])
            out_img = self.img_tensor2numpy(out_img)
            out_imgs.append(out_img)
        out_imgs = np.concatenate(out_imgs, axis=0)

        return out_imgs

    def get_sparse_point(self, model_preds, lp_canonical_kp):
        T = model_preds.shape[0]

        exp, R, t, scale = \
            model_preds[:, :63], model_preds[:, 63:66], \
            model_preds[:, 66:69].unsqueeze(1), model_preds[:, 69:].unsqueeze(1)
        exp = exp.reshape(-1, 21, 3)
        R = get_rotation_matrix(R[:, 0], R[:, 1], R[:, 2])
        t[..., 2].fill_(0)  # zero tz
        x = scale * (lp_canonical_kp.repeat(T, 1, 1) @ R + exp) + t  # (T, 21, 3)

        return x

    def get_audio_embedding(self, audio_path):
        audio_embs = self.audio_processor.preprocess(audio_path)[0].unsqueeze(0).to(self.device)
        video_length = audio_embs.shape[1]
        suffix_len = self.n_motion - (video_length % self.n_motion)
        audio_embs = F.pad(audio_embs, (0, 0, 0, 0, self.prev_n_motion, suffix_len), mode='replicate')
        prev_audio_embs = audio_embs[:, :self.prev_n_motion]
        audio_embs = audio_embs[:, self.prev_n_motion:]

        return prev_audio_embs, audio_embs, video_length

    def get_emotion_embedding(self, emotion_path):
        motion_embs = torch.load(emotion_path, weights_only=True).to(self.device) * 1.0
        if motion_embs.shape[1] < 11:
            motion_embs = torch.cat([motion_embs, torch.zeros((1, 11 - motion_embs.shape[1])).to(self.device)], dim=-1)
        # # 'anger', 'disgust', 'contempt', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'anxiety', 'helplessness', 'disappointment'
        # motion_embs = torch.tensor([[0, 0, 0, 0, 1.9, 0, 0.0, 0, 0, 0, 0]]).to(device)

        return motion_embs

    @torch.no_grad()
    def inference(
        self,
        image_path,
        audio_path,
        save_path,
        emotion_path,
        pose_driven_video_path,
        crop_face=False,
        no_lipoffset=False,
        rotate=False,
        fix_pose=False,
    ):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('ffmpeg not found in PATH. Install ffmpeg first (see README).')

        prev_audio_embs, audio_embs, video_length = self.get_audio_embedding(audio_path)
        motion_embs = self.get_emotion_embedding(emotion_path)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # composite RGBA onto a white background
            alpha = img[..., 3:4].astype(np.float32) / 255.0
            img = (img[..., :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        img = img[..., ::-1]  # BGR -> RGB
        if crop_face:
            crop_result = self.cropper.crop_source_image(img, self.crop_cfg)
            if crop_result is None:
                raise RuntimeError(
                    f'No face detected in {image_path}; '
                    'retry without --crop_face or use an image with a visible face.'
                )
            img = crop_result['img_crop_256x256']
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        f_s, canonical_lp_motion, lp_canonical_kp = self.get_img_lp_motion(img)
        lp_canonical_kps = lp_canonical_kp.reshape(1, 1, 1, -1)
        canonical_lp_motions = self.normalize_standard(canonical_lp_motion.reshape(1, 1, -1)).reshape(1, 1, 1, -1)
        canonical_lp_motions[..., 63:] = 0.0

        prev_lp_motions = canonical_lp_motions.clone()
        prev_lp_motions = prev_lp_motions.repeat(1, self.prev_n_motion, 1, 1)[..., lp_motion_idxs]

        bs = 50
        model_preds = []
        for mb_idx in tqdm(range(0, audio_embs.shape[1], bs)):
            model_pred = self.a2f_model.ddpm_sample(
                audio_embs[:, mb_idx:mb_idx + bs],
                motion_embs,
                prev_lp_motions,
                prev_audio_embs,
                canonical_lp_motions[..., lp_motion_idxs],
                cfg_scale=self.cfg_scale,
                target_mode='noise',
                add_style=True
            )
            prev_lp_motions = model_pred[:, -self.prev_n_motion:]
            prev_audio_embs = audio_embs[:, mb_idx + bs - self.prev_n_motion:mb_idx + bs]
            model_preds.append(model_pred)
        model_preds = torch.cat(model_preds, dim=1)[0, :video_length]

        T = model_preds.shape[0]

        tmp_model_preds = canonical_lp_motions[0].repeat(T, 1, 1)
        tmp_model_preds[..., lp_motion_idxs] = model_preds
        model_preds = tmp_model_preds.clone()
        model_preds[..., :63] = self.reverse_normalize_standard(tmp_model_preds)[..., :63]

        model_preds, rotate_preds, fix_pose_preds = self.motion_modulation(
            model_preds, canonical_lp_motion, no_lipoffset, pose_driven_video_path,
            rotate=rotate, fix_pose=fix_pose,
        )

        lp_temp_motion = canonical_lp_motion.clone().reshape(1, -1)
        lp_canonical_kp = lp_canonical_kps[0].reshape(1, -1, 3)
        x_c = self.get_sparse_point(lp_temp_motion, lp_canonical_kp)
        x_c = x_c.repeat(T, 1, 1)  # (hT, 21, 3)

        f_s = f_s.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        if rotate and fix_pose:
            x_d = self.get_sparse_point(rotate_preds[:, 0, :], lp_canonical_kp)
            rotate_imgs = self.warp_and_generate(x_c, x_d, f_s)

            x_d = self.get_sparse_point(fix_pose_preds[:, 0, :], lp_canonical_kp)
            fix_pose_imgs = self.warp_and_generate(x_c, x_d, f_s)

            out_imgs = np.concatenate([fix_pose_imgs, rotate_imgs], axis=1)
        else:
            x_d = self.get_sparse_point(model_preds[:, 0, :], lp_canonical_kp)
            out_imgs = self.warp_and_generate(x_c, x_d, f_s)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        silence_save_path = add_suffix_name(save_path, '_silence')
        try:
            images2video(out_imgs, silence_save_path, fps=25)
            add_audio_to_video(silence_save_path, audio_path, save_path)
        finally:
            if os.path.exists(silence_save_path):
                os.remove(silence_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--emotion_path", type=str,
                        default=os.path.join(BASE_DIR, 'data', 'onehot_emotions', 'neutral.pt'))
    parser.add_argument('--pose_driven_video_path', type=str, default=None)
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='root directory of pretrained weights (default: <repo>/pretrained_weights)')
    parser.add_argument('--device', type=str, default=None,
                        help='e.g. cuda, cuda:0, cpu (default: auto)')
    parser.add_argument('--crop_face', action="store_true", default=False)
    parser.add_argument('--no_lipoffset', action="store_true", default=False)
    parser.add_argument('--rotate', action="store_true", default=False)
    parser.add_argument('--fix_pose', action="store_true", default=False)

    opt = parser.parse_args()
    image_path = opt.image_path
    audio_path = opt.audio_path
    save_path = opt.save_path
    emotion_path = opt.emotion_path
    pose_driven_video_path = opt.pose_driven_video_path
    crop_face = opt.crop_face
    no_lipoffset = opt.no_lipoffset
    rotate = opt.rotate
    fix_pose = opt.fix_pose

    pipeline = PlaymatePipeline(weights_dir=opt.weights_dir, device=opt.device)
    pipeline.inference(
        image_path,
        audio_path,
        save_path,
        emotion_path,
        pose_driven_video_path,
        crop_face=crop_face,
        no_lipoffset=no_lipoffset,
        rotate=rotate,
        fix_pose=fix_pose,
    )

