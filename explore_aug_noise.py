"""
TOOD:
- First: Load audio
- Need to store in Array
- Randomly Grab a single audio file
- Need to random crop a single portion of range len(input), if len input bigger than example need to loop, for now just crop


# Need to make sure no clipping
# Need to code out gate for when it fires: 0.8% of audio should have noise mixed in

"""

from pathlib import Path
import soundfile as sf
import random
import torchaudio
import numpy as np
import torch

PATH_TO_RANDOM_NOISE = "/Users/cameronolson/Developer/Personal/Learning/JobPrep/AudioResearcher/SC/e2e9/data/SpeechCommands/speech_commands_v0.02/_background_noise_"


# class RandomNoiseMixIn:
#     def __init__(self, cfg: Config):
#         self.path = Path(noise_path)
#         self.data: list[tuple[torch.Tensor, int]] = self.load_data()

#     def load_data(self) -> list[tuple[torch.Tensor, int]]:
#         data = []
#         for file in self.path.rglob("*.wav"):
#             if not file.exists() or file.is_dir():
#                 continue
#             audio, sr = torchaudio.load(file)
#             data.append((audio, sr))
#         return data


if __name__ == "__main__":
    file_to_augment = "/Users/cameronolson/Developer/Personal/Learning/JobPrep/AudioResearcher/SC/e2e9/data/Cat.wav"
    input_file, inp_sr = torchaudio.load(file_to_augment)

    path = Path(PATH_TO_RANDOM_NOISE)

    data: torch.Tensor = []

    for f in path.rglob("*.wav"):
        audio, sr = torchaudio.load(f)
        data.append((audio, sr))

    mix_in_file, mix_in_sr = random.choice(data)
    mix_in_file = mix_in_file[:, : mix_in_sr * 2]  # 5 seconds to test wrapping

    if inp_sr != mix_in_sr:
        input_file = torchaudio.functional.resample(input_file, inp_sr, mix_in_sr)

    input_file = input_file.numpy()

    if mix_in_file.shape[1] < input_file.shape[1]:
        needed_len = input_file.shape[1] - mix_in_file.shape[1]
        additional_files = [mix_in_file]
        while needed_len >= 0:
            file_chunk = mix_in_file[:, :needed_len]
            additional_files.append(file_chunk)
            needed_len -= mix_in_file.shape[1]

        cropped_mix_in = np.concatenate(additional_files, axis=1)
        breakpoint()
        print("Cropped it")
    else:
        crop_dur = input_file.shape[1]

        max_start = mix_in_file.shape[1] - crop_dur
        start_loc = random.randint(0, max_start)
        end_loc = start_loc + crop_dur

        cropped_mix_in = mix_in_file[:, start_loc:end_loc].numpy()

    snr = random.uniform(-5.0, 25.0)

    cropped_mix_in_power = np.mean(cropped_mix_in**2)
    input_file_power = np.mean(input_file**2)

    target_power = input_file_power / (10 ** (snr / 10.0))

    gain_factor = np.sqrt(target_power / cropped_mix_in_power)
    mixed_samples = input_file + cropped_mix_in * gain_factor
    # sf.write("test.wav", mixed_samples, mix_in_sr)
    sf.write("test_file.wav", mixed_samples.reshape(-1), mix_in_sr)
