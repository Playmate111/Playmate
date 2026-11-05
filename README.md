<h1 align="center">Playmate: Flexible Control of Portrait Animation via 3D-Implicit Space Guided Diffusion</h1>

<div align='center'>
    <a href='https://github.com/XPMarcus' target='_blank'>Xingpei Ma</a><sup>*</sup>&emsp;
    <a href='https://github.com/caijiaran1993' target='_blank'>Jiaran Cai</a><sup>*†</sup>&emsp;
    <a href='https://github.com/Guanyuansheng' target='_blank'>Yuansheng Guan</a><sup>*</sup>&emsp;
    <a href='https://github.com/huangshenneng' target='_blank'>Shenneng Huang</a><sup></sup>&emsp;
    <a href='https://github.com/' target='_blank'>Qiang Zhang</a><sup></sup>
    <a href='https://github.com/' target='_blank'>Shunsi Zhang</a><sup></sup>
</div>
<div align='center'>
    <small><sup>*</sup> Equal contribution</small>
    <small><sup>†</sup>Project lead</small>
</div>

<div align='center'>
Guangzhou Quwan Network Technology
</div>
<br>

<div align="center">
  <a href='https://arxiv.org/abs/2502.07203'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href='https://playmate111.github.io/Playmate/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
</div>

<div align='Center'>
    <i><strong><a href='https://icml.cc/Conferences/2025' target='_blank'>ICML 2025</a></strong></i>
</div>
<br>


## 📰 News
- **`2025/11/11`**: 🎉🎉🎉 Our [Playmate2 paper](https://arxiv.org/abs/2510.12089) has been accepted and will be presented at [AAAI 2026](https://aaai.org/conference/aaai/aaai-26/). We plan to release the inference code and model weights for both [**Playmate**](https://github.com/Playmate111/Playmate) and [**Playmate2**](https://github.com/Playmate111/Playmate2) in the coming weeks. Stay tuned and thank you for your patience!
- **`2025/10/15`**: 🔥🔥🔥 We released [**Playmate2**](https://github.com/Playmate111/Playmate2), a novel DiT framework for generating long-duration high-quality audio-driven videos. Playmate2 also supports multi-character animation. To the best of our knowledge, Playmate2 is the first training-free approach capable of enabling audio-driven animation for three or more characters. Codes and models will release, please stay tuned!
- **`2025/05/07`**: 🎉🎉🎉 Super stoked to share that our paper has been accepted to [ICML 2025](https://icml.cc/Conferences/2025)!
- **`2025/04/28`**: ✨✨✨ Created a GitHub repository for the project.
- **`2025/02/11`**: 🚀🚀🚀 Our [paper](https://arxiv.org/abs/2502.07203) is in public on arxiv.



## 📑 Todo List
- [x] Inference code for the first stage.
- [x] Release pretrained models for the first stage.
- [ ] Inference code for the second stage.
- [ ] Release pretrained models for the second stage.
- [ ] Release the training code for the first stage.
- [ ] Release the training code for the second stage.



## 📸 Showcase
### Audio Driven (Talking)

<table class="center">
    
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/e6058adf-9a0c-430f-b00a-97ff1c2029fe" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/505d1c5d-5158-4221-bf33-c823083d46e5" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/00c407dd-5d88-400d-a5f4-d139960a72af" muted="false"></video>
    </td>
</tr>

</table>


### Audio Driven (Sing)

<table class="center">
    
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/e53f9afa-82aa-42a9-b0a9-30ce5aff83f0" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/284afe10-a95d-4aef-87fe-89a89c8b1052" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/99457b6b-49dc-4709-b0e6-c58e63f3ecc2" muted="false"></video>
    </td>
</tr>

</table>


### Emotion Controllability
<table> 
    <tr>
        <th style="width:14.28%;">Angry</th>
        <th style="width:14.28%;">Disgusted</th>
        <th style="width:14.28%;">Contempt</th>
        <th style="width:14.28%;">Fear</th>
        <th style="width:14.28%;">Happy</th>
        <th style="width:14.28%;">Sad</th>
        <th style="width:14.28%;">Surprised</th>
    </tr>
    <tr>
        <td colspan="7">
            <video controls loop src="https://github.com/user-attachments/assets/55119d12-8a21-434b-8ea0-8cdecc17d99d" muted="false"></video>
        </td>
    </tr>
    <tr>
        <td colspan="7">
            <video controls loop src="https://github.com/user-attachments/assets/899507f0-fbe1-4ce8-9b57-f9d2105c3d64" muted="false"></video>
        </td>
    </tr>
    <tr>
        <td colspan="7">
            <video controls loop src="https://github.com/user-attachments/assets/79181db1-bae6-42d4-8308-ac33ef85751a" muted="false"></video>
        </td>
    </tr>
    <tr>
        <td colspan="7">
            <video controls loop src="https://github.com/user-attachments/assets/2c761809-ea64-477b-bb5d-0c99e179ebdf" muted="false"></video>
        </td>
    </tr>
    <tr>
        <td colspan="7">
            <video controls loop src="https://github.com/user-attachments/assets/af2b68f5-1096-4301-aaf7-c3171b84393c" muted="false"></video>
        </td>
    </tr>
</table>

Explore [more examples](https://playmate111.github.io/Playmate/).


## Quick Start

> **Scope**: This repository contains the **inference code for the first stage only** (single portrait image + audio → talking/singing head video). Training code and the second-stage (long-duration / multi-character) code are **not** included.

### 🛠️Installation

**Requirements**

- Linux or Windows, Python 3.10+
- NVIDIA GPU with CUDA 12.x is recommended. `--crop_face` currently requires CUDA because the face cropper is created with CUDA execution providers.
- `ffmpeg` must be available in `PATH` (used to export audio and mux the final video).

Create conda environment:
```bash
conda create -n playmate python=3.10
conda activate playmate
```

Install packages with `pip`
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "huggingface_hub[cli]"
```

`ffmpeg` is also needed:
```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
# Windows (choose one, then reopen the terminal)
winget install Gyan.FFmpeg
# or: choco install ffmpeg
```

Verify the installation:
```bash
ffmpeg -version
```

> `requirements.txt` installs `onnxruntime-gpu`. On a CPU-only machine, replace it with `onnxruntime` manually; automatic face cropping (`--crop_face`) is not supported in that case.

### 🧱Model Preparation

#### Model Download

| Models               | Download Link                                                                                                                                                                       | Save Path                                                   |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| wav2vec2-base-960h   | [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)                                                                                                                     | `pretrained_weights/wav2vec2-base-960h`                     |
| Our Pretrained Model | [ModelScope](https://modelscope.cn/models/alkdjfklaj/Playmate)                                                                                                                         | `pretrained_weights`                                        |

Download models using `modelscope` (for our pretrained model) or `huggingface-cli` (for wav2vec2):

**ModelScope (our pretrained model):**
```bash
pip install modelscope
modelscope download --model alkdjfklaj/Playmate --local_dir ./pretrained_weights
```

**Huggingface (wav2vec2 only):**
```bash
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_weights/wav2vec2-base-960h
```

If your `huggingface_hub` version has removed `huggingface-cli`, use the equivalent `hf download` command instead.

#### Expected directory layout

After downloading, `pretrained_weights` should look like this:

```text
pretrained_weights/
├── base_models/
│   ├── appearance_feature_extractor.pth      # required
│   ├── motion_extractor.pth                  # required
│   ├── spade_generator.pth                   # required
│   └── warping_module.pth                    # required
├── playmate/
│   └── playmate.pt                           # required
├── wav2vec2-base-960h/                       # required
├── audio_separator/
│   └── Kim_Vocal_2.onnx                      # required (vocal separation)
├── insightface/
│   └── models/
│       └── buffalo_l/                        # only for --crop_face
└── landmark.onnx                             # only for --crop_face
```

- Weights are not tracked by git. Use `--weights_dir` if you store them elsewhere.
- `buffalo_l` is downloaded automatically from GitHub on the first `--crop_face` run if it is missing (network required). Pre-place it under `pretrained_weights/insightface/models/buffalo_l/` to avoid the download.
- `landmark.onnx` is only needed for `--crop_face`; it cannot be downloaded automatically.

### 🎮Inference

```bash
python a2f_inference.py \
  --image_path examples/images/01.png \
  --audio_path examples/audios/01.wav \
  --save_path examples/outputs/01_01.mp4 \
  --emotion_path data/onehot_emotions/neutral.pt
```

The result is a **512×512, 25 fps MP4** with the original audio muxed in. The parent directory of `--save_path` is created automatically.

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_path` | required | Source portrait image. A square image with a centered face works best. |
| `--audio_path` | required | Driving audio (talking or singing). |
| `--save_path` | required | Output MP4 path, e.g. `examples/outputs/01_01.mp4`. |
| `--emotion_path` | `data/onehot_emotions/neutral.pt` | Emotion one-hot encoding. See the emotion table below. |
| `--pose_driven_video_path` | `None` | Optional pose driver. Accepts a `.mp4` video (must be exactly 25 fps) or a single image. |
| `--weights_dir` | `<repo>/pretrained_weights` | Root directory of the pretrained weights. |
| `--device` | auto (`cuda` if available, else `cpu`) | Inference device, e.g. `cuda`, `cuda:0`, `cpu`. |
| `--crop_face` | `False` | Automatically detect and crop the face. Requires CUDA and the `insightface` / `landmark.onnx` assets. |
| `--no_lipoffset` | `False` | Disable the lip-offset correction applied to predicted expression motion. |
| `--rotate` | `False` | Apply a horizontal head-rotation sweep (−30° to +30°) to the output. |
| `--fix_pose` | `False` | Keep the head pose fixed while animating the expression. |

#### Emotion styles

`data/onehot_emotions` contains 8 categories: 7 emotions with 3 intensity levels each, plus neutral.

| Category | Files |
|----------|-------|
| Angry | `angry_level_1.pt`, `angry_level_2.pt`, `angry_level_3.pt` |
| Disgusted | `disgusted_level_1.pt`, `disgusted_level_2.pt`, `disgusted_level_3.pt` |
| Contempt | `contempt_level_1.pt`, `contempt_level_2.pt`, `contempt_level_3.pt` |
| Fear | `fear_level_1.pt`, `fear_level_2.pt`, `fear_level_3.pt` |
| Happy | `happy_level_1.pt`, `happy_level_2.pt`, `happy_level_3.pt` |
| Sad | `sad_level_1.pt`, `sad_level_2.pt`, `sad_level_3.pt` |
| Surprised | `surprised_level_1.pt`, `surprised_level_2.pt`, `surprised_level_3.pt` |
| Neutral | `neutral.pt` |

#### More examples

```bash
# Automatic face cropping
python a2f_inference.py \
  --image_path examples/images/02.png \
  --audio_path examples/audios/01.wav \
  --save_path examples/outputs/02_crop.mp4 \
  --crop_face

# A specific emotion intensity
python a2f_inference.py \
  --image_path examples/images/01.png \
  --audio_path examples/audios/01.wav \
  --save_path examples/outputs/01_happy.mp4 \
  --emotion_path data/onehot_emotions/happy_level_2.pt

# Pose driven by a 25 fps video (or a single image)
python a2f_inference.py \
  --image_path examples/images/01.png \
  --audio_path examples/audios/01.wav \
  --save_path examples/outputs/01_pose.mp4 \
  --pose_driven_video_path path/to/driving.mp4

# Head rotation and fixed pose side by side
python a2f_inference.py \
  --image_path examples/images/01.png \
  --audio_path examples/audios/01.wav \
  --save_path examples/outputs/01_rotate_fix.mp4 \
  --rotate --fix_pose
```

> With `--rotate --fix_pose` together, the output contains two videos side by side (fixed pose on the left, rotation sweep on the right).


## 🗂️ Repository Structure

```text
.
├── a2f_inference.py              # inference entry point + PlaymatePipeline
├── config/
│   ├── models.yaml               # architecture hyper-parameters
│   ├── a2f_diffusion_config.py   # audio-to-motion diffusion config
│   ├── crop_config.py            # face cropping config
│   └── base_config.py
├── modules/
│   ├── a2f_net.py                # diffusion wrapper (A2FDiffNet)
│   ├── a2f_diffusion_modules/    # denoising decoder / conformer / DiT blocks
│   ├── motion_extractor.py       # 3D implicit keypoint extraction
│   ├── appearance_feature_extractor.py
│   ├── warping_network.py        # feature warping
│   ├── dense_motion.py
│   ├── spade_generator.py        # SPADE decoder (256 → 512)
│   └── ...
├── utils/                        # audio, video, cropping, visualization helpers
├── data/
│   ├── onehot_emotions/          # emotion style encodings
│   └── lp_motion_*.pt            # motion normalization statistics
├── examples/                     # sample portrait image and audio
├── pretrained_weights/           # weights (not tracked by git)
├── requirements.txt
└── LICENSE
```

### Inference pipeline

1. **Audio processing** – vocal separation (`Kim_Vocal_2.onnx`), then Wav2Vec2 features at 25 fps.
2. **Audio-to-motion diffusion** – `A2FDiffNet` predicts a 52-d motion sequence conditioned on audio, emotion, previous motions and the source canonical motion.
3. **3D keypoints** – the predicted motion is combined with the canonical 21 keypoints to obtain per-frame 3D keypoints.
4. **Rendering** – `WarpingNetwork` warps the source feature volume and `SPADEDecoder` renders 512×512 frames; `ffmpeg` then muxes the audio.


## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Missing required files: ...` | Check the expected `pretrained_weights` layout above, or pass `--weights_dir /path/to/weights`. |
| `ffmpeg not found in PATH` | Install `ffmpeg` and reopen the terminal; verify with `ffmpeg -version`. |
| `--crop_face` fails with a CUDA / `CUDAExecutionProvider` error | The cropper requires CUDA and `onnxruntime-gpu`. Run on a CUDA machine, install the GPU build of `onnxruntime`, or omit `--crop_face`. |
| `No face detected in ...` | Use a clear, front-facing portrait with a visible face, or run without `--crop_face`. |
| `OSError`/`Can't load ... wav2vec2-base-960h` | Download the full `wav2vec2-base-960h` folder into `pretrained_weights/wav2vec2-base-960h`; the model is loaded with `local_files_only=True`. |
| `Driven video fps is not equal to 25 fps` | Re-encode the pose-driving video to exactly 25 fps, or use a single image as the pose driver. |
| Inference is very slow / out of memory | The denoising sampler runs 1000 steps. Use a CUDA GPU; reduce the audio length or free GPU memory. |
| First run is slower than later runs | The first `--crop_face` run may download `buffalo_l`, and vocal separation is performed per audio file. |


## 📄 License

This project is released under the [Apache License 2.0](LICENSE).


## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@inproceedings{maplaymate,
  title={Playmate: Flexible Control of Portrait Animation via 3D-Implicit Space Guided Diffusion},
  author={Ma, Xingpei and Cai, Jiaran and Guan, Yuansheng and Huang, Shenneng and Zhang, Qiang and Zhang, Shunsi},
  booktitle={Forty-second International Conference on Machine Learning}
}

```
