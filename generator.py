import os

from imgaug import augmenters as iaa
import imgaug as ia
from keras.utils import Sequence
import numpy as np


class SiameseSequence(Sequence):
    def __init__(self,
                 path: str,
                 stage: str = "train",
                 batch_size: int = 128,
                 ):

        self.lazy_loaded = False

        self.images = []
        self.images_filenames = []
        self.labels = []

        for r, d, f in os.walk(os.path.join(path, "images")):
            for file in f:
                self.images.append(os.path.join(r, file))
                self.labels.append(os.path.join(path, "labels", (file.split(".")[0] + ".txt")))

        self.batch_size = batch_size
        self.stage = stage

        self.seq = SiameseSequence.create_augmenter(stage)

    def __len__(self):
        return int(np.floor(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        global cv2

        if not self.lazy_loaded:
            import cv2
            self.lazy_loaded = True

        batch_input_left = np.zeros((self.batch_size, 224, 224, 3))
        batch_input_right = np.zeros((self.batch_size, 224, 224, 3))

        seq_det = self.seq.to_deterministic()

        for i in range(0, self.batch_size):

            image = cv2.imread(self.images[idx * self.batch_size + i])
            bboxes = SiameseSequence.load_kitti_label(image,
                                                      scale=(image.shape[0],
                                                             image.shape[1]),
                                                      label=self.labels[idx * self.batch_size + i])

            # Each bounding box is a training example
            for box in bboxes.to_xyxy_array():
                x1, y1, x2, y2 = box

                center = (np.array([x1, y1]) + np.array([x2, y2])) / 2



        batch_output = np.zeros(self.batch_size, 4)

        return [batch_input_left, batch_input_right], batch_output

    @staticmethod
    # KITTI Format Labels
    def load_kitti_label(image: np.ndarray, scale, label: str):

        label = open(label, 'r').read().strip()

        bboxes = []

        for row in label.split('\n'):
            fields = row.split(' ')

            bbox_class = fields[0]

            bbox_x1 = float(fields[4]) * scale[1]
            bbox_y1 = float(fields[5]) * scale[0]
            bbox_x2 = float(fields[6]) * scale[1]
            bbox_y2 = float(fields[7]) * scale[0]

            bbox = ia.BoundingBox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_class)
            bboxes.append(bbox)

        bboi = ia.BoundingBoxesOnImage(bboxes, shape=image.shape)

        return bboi

    @staticmethod
    def create_augmenter(stage: str = "train"):
        if stage == "train":
            return iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.CropAndPad(px=(0, 112), sample_independently=False),
                iaa.Affine(translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}),
                iaa.SomeOf((0, 3), [
                    iaa.AddToHueAndSaturation((-10, 10)),
                    iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    iaa.AdditiveGaussianNoise(scale=0.05 * 255)
                ])
            ])
        elif stage == "val":
            return iaa.Sequential([
                iaa.CropAndPad(px=(0, 112), sample_independently=False),
                iaa.Affine(translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}),
            ])
        elif stage == "test":
            return iaa.Sequential([])
