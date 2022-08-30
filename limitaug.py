import os
from glob import glob
import random
import math
from typing import Optional, Callable

import numpy as np
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
import pyloudnorm as pyln
from pedalboard import Pedalboard, Limiter, Gain


# based on https://github.com/sigsep/open-unmix-pytorch
def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


# numpy based augmentation
# based on https://github.com/sigsep/open-unmix-pytorch
def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + random.random() * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and random.random() < 0.5:
        return np.flip(audio, axis=0)  # axis=0 must be given
    else:
        return audio


# load wav file from arbitrary positions of 16bit stereo wav file
def load_wav_arbitrary_position_stereo(filename, sample_rate, seq_duration):
    # stereo
    # seq_duration[second]
    length = torchaudio.info(filename).num_frames

    random_start = random.randint(
        0, int(length - math.ceil(seq_duration * sample_rate) - 1)
    )
    random_start_sec = librosa.samples_to_time(random_start, sr=sample_rate)
    X, sr = librosa.load(
        filename, sr=None, mono=False, offset=random_start_sec, duration=seq_duration
    )

    return X


# Linear gain increasing implementation for Method (1)
def apply_linear_gain_increase(mixture, target, board, meter, samplerate, target_lufs):
    mixture, target = mixture.T, target.T
    loudness = meter.integrated_loudness(mixture)

    if np.isinf(loudness):
        augmented_gain = 0.0
        board[0].gain_db = augmented_gain
    else:
        augmented_gain = target_lufs - loudness
        board[0].gain_db = augmented_gain
    mixture = board(mixture.T, samplerate)
    target = board(target.T, samplerate)
    return mixture, target


# LimitAug implementation for Method (2)
def apply_limitaug(audio, board, meter, samplerate, target_lufs):
    audio = audio.T
    loudness = meter.integrated_loudness(audio)

    if np.isinf(loudness):
        augmented_gain = 0.0
        board[0].gain_db = augmented_gain
    else:
        augmented_gain = target_lufs - loudness
        board[0].gain_db = augmented_gain
    audio = board(audio.T, samplerate)
    return audio


# Implementation of LimitAug then Loudness normalization  for Method (4)
def apply_limitaug_loudnorm(
    audio, board, meter, samplerate, target_lufs, target_loudnorm_lufs=-14.0
):
    audio = audio.T
    loudness = meter.integrated_loudness(audio)

    if np.isinf(loudness):
        augmented_gain = 0.0
        board[0].gain_db = augmented_gain
    else:
        augmented_gain = target_lufs - loudness
        board[0].gain_db = augmented_gain
    audio = board(audio.T, samplerate)

    after_loudness = meter.integrated_loudness(audio.T)

    if np.isinf(after_loudness):
        pass
    else:
        target_gain = target_loudnorm_lufs - after_loudness
        audio = audio * db2linear(target_gain)

    return audio


def linear2db(x, eps=1e-5, scale=20):
    return scale * np.log10(x + eps)


def db2linear(x, eps=1e-5, scale=20):
    return 10 ** (x / scale) - eps


"""
This dataloader implementation is based on https://github.com/sigsep/open-unmix-pytorch
"""
class musdb_train_Dataset(Dataset):
    def __init__(
        self,
        target: str = "vocals",
        root: str = None,
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        sample_rate: int = 44100,
        seed: int = 42,
        limitaug_method: str = "limitaug_then_loudnorm",
        limitaug_mode: str = "normal_L",
        limitaug_custom_target_lufs: float = None,
        target_loudnorm_lufs: float = -14.0,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        limitaug_method : str
        choose from ["linear_gain_increase", "limitaug", "limitaug_then_loudnorm", "only_loudnorm"]

        limitaug_mode : str
        choose from ["uniform", "normal", "normal_L", "normal_XL", "normal_short_term", "normal_L_short_term", "normal_XL_short_term", "custom"]

        limitaug_custom_target_lufs : float
        valid only when
        limitaug_mode == "custom"

        target_loudnorm_lufs : float
        valid only when
        limitaug_method == 'limitaug_then_loudnorm' or 'only_loudnorm'
        default is -14.
        To the best of my knowledge, Spotify and Youtube music is using -14 as a reference loudness normalization level.
        No special reason for the choice of -14 as target_loudnorm_lufs.

        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.
        """

        self.seed = seed
        random.seed(seed)
        self.seq_duration = seq_duration
        self.target = target
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.sample_rate = sample_rate

        self.root = root
        self.sources = ["vocals", "bass", "drums", "other"]
        self.train_list = glob(f"{self.root}/train/*")
        self.valid_list = [
            "ANiMAL - Rockshow",
            "Actions - One Minute Smile",
            "Alexander Ross - Goodbye Bolero",
            "Clara Berry And Wooldog - Waltz For My Victims",
            "Fergessen - Nos Palpitants",
            "James May - On The Line",
            "Johnny Lokke - Promises & Lies",
            "Leaf - Summerghost",
            "Meaxic - Take A Step",
            "Patrick Talbot - A Reason To Leave",
            "Skelpolu - Human Mistakes",
            "Traffic Experiment - Sirens",
            "Triviul - Angelsaint",
            "Young Griffo - Pennies",
        ]

        self.train_list = [
            x for x in self.train_list if os.path.basename(x) not in self.valid_list
        ]

        # limitaug related
        self.limitaug_method = limitaug_method
        self.limitaug_mode = limitaug_mode
        self.limitaug_custom_target_lufs = limitaug_custom_target_lufs
        self.target_loudnorm_lufs = target_loudnorm_lufs
        self.meter = pyln.Meter(self.sample_rate)

        # Method (1) in our paper's Results section and Table 5
        if self.limitaug_method == "linear_gain_increase":
            print("using linear gain increasing!")
            self.board = Pedalboard([Gain(gain_db=0.0)])

        # Method (2) in our paper's Results section and Table 5
        elif self.limitaug_method == "limitaug":
            print("using limitaug!")
            self.board = Pedalboard(
                [Gain(gain_db=0.0), Limiter(threshold_db=0.0, release_ms=100.0)]
            )

        # Method (3) in our paper's Results section and Table 5
        elif self.limitaug_method == "only_loudnorm":
            print("using only loudness normalized inputs")

        # Method (4) in our paper's Results section and Table 5
        elif self.limitaug_method == "limitaug_then_loudnorm":
            print("using limitaug then loudness normalize!")
            self.board = Pedalboard(
                [Gain(gain_db=0.0), Limiter(threshold_db=0.0, release_ms=100.0)]
            )

        self.limitaug_mode_statistics = {
            "normal": [
                -15.954,
                1.264,
            ],  # -15.954 is mean LUFS of musdb-hq and 1.264 is standard deviation
            "normal_L": [
                -10.887,
                1.191,
            ],  # -10.887 is mean LUFS of musdb-L and 1.191 is standard deviation
            "normal_XL": [
                -8.608,
                1.165,
            ],  # -8.608 is mean LUFS of musdb-L and 1.165 is standard deviation
            "normal_short_term": [
                -17.317,
                5.036,
            ],  # In our experiments, short-term statistics were not helpful.
            "normal_L_short_term": [-12.303, 5.233],
            "normal_XL_short_term": [-9.988, 5.518],
            "custom": [limitaug_custom_target_lufs, 1.2],
        }

    def get_limitaug_results(self, mixture, target):
        # Apply linear gain increasing (Method (1))
        if self.limitaug_method == "linear_gain_increase":
            if (
                self.limitaug_mode == "uniform"
            ):  # if limitaug_mode is uniform, then choose target_lufs from uniform distribution
                target_lufs = random.uniform(-20, -5)
            else:  # else, choose target_lufs from gaussian distribution
                target_lufs = random.gauss(
                    self.limitaug_mode_statistics[self.limitaug_mode][0],
                    self.limitaug_mode_statistics[self.limitaug_mode][1],
                )
            mixture, target = apply_linear_gain_increase(
                mixture,
                target,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=target_lufs,
            )

        # Apply LimitAug (Method (2))
        elif self.limitaug_method == "limitaug":
            self.board[1].release_ms = random.uniform(30.0, 200.0)
            mixture_orig = mixture.copy()
            if (
                self.limitaug_mode == "uniform"
            ):  # if limitaug_mode is uniform, then choose target_lufs from uniform distribution
                target_lufs = random.uniform(-20, -5)
            else:  # else, choose target_lufs from gaussian distribution
                target_lufs = random.gauss(
                    self.limitaug_mode_statistics[self.limitaug_mode][0],
                    self.limitaug_mode_statistics[self.limitaug_mode][1],
                )
            mixture = apply_limitaug(
                mixture,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=target_lufs,
            )
            target *= mixture / (mixture_orig + 1e-8)

        # Apply only loudness normalization (Method(3))
        elif self.limitaug_method == "only_loudnorm":
            mixture_loudness = self.meter.integrated_loudness(mixture.T)
            if np.isinf(
                mixture_loudness
            ):  # if the source is silence, then mixture_loudness is -inf.
                pass
            else:
                augmented_gain = (
                    self.target_loudnorm_lufs - mixture_loudness
                )  # default target_loudnorm_lufs is -14.
                mixture = mixture * db2linear(augmented_gain)
                target = target * db2linear(augmented_gain)

        # Apply LimitAug then loudness normalization (Method (4))
        elif self.limitaug_method == "limitaug_then_loudnorm":
            self.board[1].release_ms = random.uniform(30.0, 200.0)
            mixture_orig = mixture.copy()
            if (
                self.limitaug_mode == "uniform"
            ):  # if limitaug_mode is uniform, then choose target_lufs from uniform distribution
                target_lufs = random.uniform(-20, -5)
            else:  # else, choose target_lufs from gaussian distribution
                target_lufs = random.gauss(
                    self.limitaug_mode_statistics[self.limitaug_mode][0],
                    self.limitaug_mode_statistics[self.limitaug_mode][1],
                )
            mixture = apply_limitaug_loudnorm(
                mixture,
                self.board,
                self.meter,
                self.sample_rate,
                target_lufs=target_lufs,
                target_loudnorm_lufs=self.target_loudnorm_lufs,
            )
            target *= mixture / (mixture_orig + 1e-8)

        return mixture, target

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        for k, source in enumerate(self.sources):
            # memorize index of target source
            if source == self.target:  # if source is 'vocals'
                target_ind = k
                track_path = self.train_list[
                    index // self.samples_per_track
                ]  # we want to use # training samples per each track.
                audio_path = f"{track_path}/{source}.wav"
                audio = load_wav_arbitrary_position_stereo(
                    audio_path, self.sample_rate, self.seq_duration
                )
            else:
                track_path = random.choice(self.train_list)
                audio_path = f"{track_path}/{source}.wav"
                audio = load_wav_arbitrary_position_stereo(
                    audio_path, self.sample_rate, self.seq_duration
                )
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = np.stack(audio_sources, axis=0)

        # # apply linear mix over source index=0
        x = stems.sum(0)
        # get the target stem
        y = stems[target_ind]

        # Apply the limitaug,
        x, y = self.get_limitaug_results(x, y)

        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.train_list) * self.samples_per_track


# If you want to check if the LUFS values of training examples, run this.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Make musdb-L and musdb-XL dataset from its ratio data"
    )

    parser.add_argument(
        "--musdb_root",
        type=str,
        default="/path/to/musdb",
        help="root path of musdb-hq dataset",
    )
    parser.add_argument(
        "--limitaug_method",
        type=str,
        default="limitaug",
        choices=[
            "linear_gain_increase",
            "limitaug",
            "limitaug_then_loudnorm",
            "only_loudnorm",
            None,
        ],
        help="choose limitaug method",
    )
    parser.add_argument(
        "--limitaug_mode",
        type=str,
        default="normal_L",
        choices=[
            "uniform",
            "normal",
            "normal_L",
            "normal_XL",
            "normal_short_term",
            "normal_L_short_term",
            "normal_XL_short_term",
            "custom",
        ],
        help="if you use LimitAug, what lufs distribution to target",
    )
    parser.add_argument(
        "--limitaug_custom_target_lufs",
        type=float,
        default=None,
        help="if limitaug_mode is custom, set custom target lufs for LimitAug",
    )

    args, _ = parser.parse_known_args()

    source_augmentations_ = aug_from_str(["gain", "channelswap"])

    train_dataset = musdb_train_Dataset(
        target="vocals",
        root=args.musdb_root,
        seq_duration=6.0,
        source_augmentations=source_augmentations_,
        limitaug_method=args.limitaug_method,
        limitaug_mode=args.limitaug_mode,
        limitaug_custom_target_lufs=args.limitaug_custom_target_lufs,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    meter = pyln.Meter(44100)
    for i in range(5):
        for x, y in dataloader:
            loudness = meter.integrated_loudness(x[0].numpy().T)
            print(f"mixture loudness : {loudness} LUFS")
