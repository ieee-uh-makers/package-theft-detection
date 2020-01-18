from imgaug import augmenters as iaa
from keras.utils import Sequence
from keras.applications.mobilenet import preprocess_input
import numpy as np
import os
import pandas as pd
import random


class ActivitySequence(Sequence):
    def __init__(self,
                 csv_path: str,
                 source_path: str,
                 timesteps: int = 5,
                 delta_t: float = 1.0,
                 stage: str = "train",
                 batch_size: int = 24
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

        self.seq = ActivitySequence.create_augmenter(stage)

        self.sequences_pos = None
        self.sequences_neg = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.sample_without_replacement()

    def sample_without_replacement(self):
        self.sequences_pos = [str(uuid) for uuid in self.df_pos['uuid'].unique()]
        self.sequences_neg = [str(uuid) for uuid in self.df_neg['uuid'].unique()]

        random.shuffle(self.sequences_pos)
        random.shuffle(self.sequences_neg)

    def __len__(self):
        return len(self.sequences_pos) + len(self.sequences_neg)

    def __getitem__(self, idx):
        global cv2

        # Lazy load OpenCV due to OpenMP issue with multiprocessing (used by fit_generator in keras)
        if not self.lazy_loaded:
            import cv2
            self.lazy_loaded = True

        batch_input = np.zeros((self.batch_size, self.timesteps, 224, 224, 3))
        batch_output = np.zeros((self.batch_size, 1))

        seq_det = self.seq.to_deterministic()

        # Validation should always be the same for each batch
        if self.stage == 'val':
            np.random.seed(0)

        i = 0
        while i < self.batch_size:

            try:
                uuid = self.sequences_pos.pop()
            except IndexError:
                self.sample_without_replacement()
                uuid = self.sequences_pos.pop()

            sequence = self.df_pos[self.df['uuid'] == uuid].iloc[0]

            fps = sequence['fps']
            frames = sequence['frames']
            theft_idx = sequence['theft_idx']

            # Positive Sample
            pos_bound_start = max(theft_idx - fps * self.delta_t * (self.timesteps - 1), 1)
            pos_bound_end = min(theft_idx, frames - fps * self.delta_t * self.timesteps)

            pos_idx_start = np.random.randint(pos_bound_start, pos_bound_end)
            pos_idx = pos_idx_start

            for idx in range(0, self.timesteps):
                img = cv2.imread(os.path.join(self.source_path, '%s_%d.png' % (uuid, pos_idx)))
                img = seq_det.augment_image(img)
                img = preprocess_input(img)

                batch_input[i, idx] = img
                batch_output[i] = 1

                pos_idx += fps * self.delta_t

            i += 1

            # Negative Sample
            if theft_idx > fps * self.delta_t * self.timesteps + 1 or np.random.random() > 0.5:
                try:
                    uuid = self.sequences_neg.pop()
                except IndexError:
                    self.sample_without_replacement()
                    uuid = self.sequences_neg.pop()

                sequence = self.df_neg[self.df['uuid'] == uuid].iloc[0]

                fps = sequence['fps']
                frames = sequence['frames']

                neg_bound_start = 1
                neg_bound_end = frames - fps * self.delta_t * self.timesteps
            else:
                neg_bound_start = 1
                neg_bound_end = theft_idx - fps * self.delta_t * self.timesteps

            try:
                neg_idx_start = np.random.randint(neg_bound_start, neg_bound_end)
            except ValueError:
                neg_idx_start = 1

            neg_idx = neg_idx_start

            for idx in range(0, self.timesteps):
                img = cv2.imread(os.path.join(self.source_path, '%s_%d.png' % (uuid, neg_idx)))
                img = seq_det.augment_image(img)
                img = preprocess_input(img)

                batch_input[i, idx] = img
                batch_output[i] = 0

                neg_idx += fps * self.delta_t

            i += 1

        return batch_input, batch_output

    @staticmethod
    def create_augmenter(stage: str = "train"):
        if stage == "train":
            return iaa.Sequential([
                iaa.Resize((224, 224), interpolation='linear'),
                iaa.Fliplr(0.5),
                iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                           rotate=(-5, 5)),
                iaa.SomeOf((0, 2), [
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    iaa.AdditiveGaussianNoise(scale=0.03 * 255)
                ])
            ], random_order=True)
        elif stage == "val":
            return iaa.Sequential([iaa.Resize((224, 224), interpolation='linear')])
        elif stage == "test":
            return iaa.Sequential([iaa.Resize((224, 224), interpolation='linear')])


if __name__ == '__main__':
    seq = ActivitySequence(csv_path='csv/train.csv', source_path='data')
    for i in range(0, len(seq)):
        ip, op = seq.__getitem__(i)
        print(i)
