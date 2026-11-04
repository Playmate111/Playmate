import math
import os
import tempfile

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from utils.wav2vec import Wav2VecModel
from utils.util import resample_audio


class AudioProcessor:
    def __init__(
        self,
        wav2vec_model_path,
        audio_separator_model_path:str=None,
        audio_separator_model_name:str=None,
        only_last_features=False,
        sample_rate=16000,
        fps=25,
        cache_dir=None,
        device=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device=device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features

        if audio_separator_model_name is not None:
            cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'playmate_vocals')
            os.makedirs(cache_dir, exist_ok=True)
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            if self.audio_separator.model_instance is None:
                raise RuntimeError("Fail to load audio separate model.")
        else:
            self.audio_separator=None
            print("Use audio directly without vocals seperator.")

        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)

    def preprocess(self, wav_file: str, 
                   clip_length: int=-1, 
                   padding=False,
                   processed_length=0):
        if self.audio_separator is not None:
            outputs = self.audio_separator.separate(wav_file)
            if len(outputs) <= 0:
                raise RuntimeError("Audio separate failed.")
            vocal_audio_file = outputs[0]
            vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
            vocal_audio_file = os.path.join(self.audio_separator.output_dir, vocal_audio_file)
            vocal_audio_file = resample_audio(vocal_audio_file, os.path.join(self.audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"), self.sample_rate)
        else:
            vocal_audio_file=wav_file

        speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=self.sample_rate)
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)
        audio_length = seq_len

        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)

        if padding:
            if clip_length>0 and seq_len % clip_length != 0:
                all_len = seq_len + processed_length
                audio_feature = torch.nn.functional.pad(audio_feature, (0, (clip_length - all_len % clip_length) * (self.sample_rate // self.fps)), 'constant', 0.0)
                seq_len += clip_length - all_len % clip_length
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
        if len(embeddings) == 0:
            raise RuntimeError("Fail to extract audio embedding")
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()

        return audio_emb, audio_length
