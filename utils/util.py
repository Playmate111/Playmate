import os
import subprocess


def add_suffix_name(path, suffix):
    split_name = os.path.basename(path).split('.')
    name, format = '.'.join(split_name[:-1]), split_name[-1]
    return os.path.join(os.path.dirname(path), name + suffix + '.' + format)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Resample audio failed (ffmpeg exit code {ret}): {input_audio_file}")
    return output_audio_file
