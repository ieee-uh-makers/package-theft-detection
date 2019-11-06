from imgaug import augmenters as iaa
from keras.utils import Sequence
from keras.applications.mobilenet import preprocess_input
import numpy as np
import os
import pandas as pd


class ActivitySequence(Sequence):
    def __init__(self,
                 csv_path: str,
                 source_path: str,
                 timesteps: int = 10,
                 delta_t: float = 1.0,
                 stage: str = "train",
                 batch_size: int = 32,
                 ):

        self.source_path = source_path

        self.lazy_loaded = False

        self.batch_size = batch_size
        self.stage = stage
        self.timesteps = timesteps
        self.delta_t = delta_t

        self.df = pd.read_csv(csv_path)
        self.df_pos = self.df[self.df['theft_idx'] != -1]
        self.df_neg = self.df[self.df['theft_idx'] == -1]

        self.sequences_pos = [str(uuid) for uuid in self.df_pos['uuid'].unique()]
        self.sequences_neg = [str(uuid) for uuid in self.df_neg['uuid'].unique()]

        self.seq = ActivitySequence.create_augmenter(stage)

    def __len__(self):
        return int(np.floor(2*len(self.sequences_pos) / float(self.batch_size)))

    def __getitem__(self, idx):
        global cv2

        # Lazy load OpenCV due to OpenMP issue with multiprocessing (used by fit_generator in keras)
        if not self.lazy_loaded:
            import cv2
            self.lazy_loaded = True

        batch_input = np.zeros((self.batch_size, self.timesteps, 225, 225, 3))
        batch_output = np.zeros((self.batch_size, 1))

        seq_det = self.seq.to_deterministic()

        i = 0
        while i < self.batch_size:

            iidx = idx * self.batch_size + i
            uuid = self.sequences_pos[iidx]

            sequence = self.df_pos[self.df['uuid'] == uuid].iloc[0]

            fps = sequence['fps']
            frames = sequence['frames']
            theft_idx = sequence['theft_idx']

            # Positive Sample
            pos_bound_start = theft_idx - fps*self.delta_t*self.timesteps
            pos_bound_end = min(theft_idx, frames - fps*self.delta_t*self.timesteps)

            pos_idx_start = np.random.randint(pos_bound_start, pos_bound_end)
            pos_idx = pos_idx_start

            for idx in range(0, self.timesteps):
                img = cv2.imread(os.path.join(self.source_path, '%s_%d.png' % (uuid, pos_idx)))
                img = seq_det.augment_image(img)
                img = preprocess_input(img)

                batch_input[i, idx] = img

                pos_idx += fps * self.delta_t

            i += 1

            # Negative Sample
            if np.random.random() > 0.5:
                # Use a completely negative sequence, otherwise use a negative part of the positive sequence
                uuid = self.sequences_neg[iidx]
                sequence = self.df_neg[self.df['uuid'] == uuid].iloc[0]

                fps = sequence['fps']
                frames = sequence['frames']
                theft_idx = sequence['theft_idx']

            neg_bound_start = 0
            neg_bound_end = theft_idx - fps*self.delta_t*self.timesteps if theft_idx != -1 else frames - fps*self.delta_t*self.timesteps

            neg_idx_start = np.random.randint(neg_bound_start, neg_bound_end)
            neg_idx = neg_idx_start

            for idx in range(0, self.timesteps):

                img = cv2.imread(os.path.join(self.source_path, '%s_%d.png' % (uuid, neg_idx)))
                img = seq_det.augment_image(img)
                img = preprocess_input(img)

                batch_input[i, idx] = img

                neg_idx += fps*self.delta_t

            i += 1

        return batch_input, batch_output

    @staticmethod
    def create_augmenter(stage: str = "train"):
        if stage == "train":
            return iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                           rotate=(-5, 5)),
                iaa.SomeOf((0, 2), [
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    iaa.AdditiveGaussianNoise(scale=0.03 * 255)
                ])
            ], random_order=True)
        elif stage == "val":
            return iaa.Sequential([])
        elif stage == "test":
            return iaa.Sequential([])
