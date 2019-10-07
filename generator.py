import os

from imgaug import augmenters as iaa
import imgaug as ia
from keras.utils import Sequence
import numpy as np
import random


class SiameseSequence(Sequence):
    def __init__(self,
                 source_path: str,
                 stage: str = "train",
                 batch_size: int = 32,
                 regr: bool = True,
                 cls: bool = False
                 ):

        self.source_path = source_path

        self.lazy_loaded = False

        self.cls = cls
        self.regr = regr

        self.images = []
        self.images_filenames = []
        self.labels = []

        for r, d, f in os.walk(os.path.join(source_path, "images")):
            for file in f:
                self.images.append(os.path.join(r, file))
                self.labels.append(os.path.join(source_path, "labels", (file.split(".")[0] + ".txt")))

        self.batch_size = batch_size
        self.stage = stage

        self.seq = SiameseSequence.create_augmenter(stage)

    def __len__(self):
        return int(np.floor(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        global cv2

        # Lazy load OpenCV due to OpenMP issue with multiprocessing (used by fit_generator in keras)
        if not self.lazy_loaded:
            import cv2
            self.lazy_loaded = True

        batch_input_static = np.zeros((self.batch_size, 224, 224, 3))
        batch_input_moving = np.zeros((self.batch_size, 224, 224, 3))
        batch_output_bbox = np.zeros((self.batch_size, 4))
        batch_output_class = np.zeros((self.batch_size, 1))

        seq_det = self.seq.to_deterministic()

        i = 0
        skip = 0
        while i - skip < self.batch_size:

            iidx = (idx * self.batch_size + i + skip) % int(self.batch_size*len(self))

            image = cv2.imread(self.images[iidx])
            if image is None:
                skip += 1
                continue

            bboxes = SiameseSequence.load_kitti_label(image, label=self.labels[iidx])

            # Online Augmentation
            image = seq_det.augment_image(image)
            bboxes = seq_det.augment_bounding_boxes(bboxes)

            bboxes_xyxy = bboxes.to_xyxy_array()

            idx = random.randrange(0, len(bboxes_xyxy))
            box = bboxes_xyxy[idx]

            bbox = bboxes.bounding_boxes[idx]

            if bbox.label == '/m/025dyy':
                batch_output_class[i] = 1
            else:
                batch_output_class[i] = 0

            siamese_images = np.zeros((2, 224, 224, 3), dtype=np.float32)

            # Original box coordinates in image
            x1, y1, x2, y2 = box.copy()

            width = x2 - x1
            height = y2 - y1

            center = np.round((np.array([x1, y1]) + np.array([x2, y2])) / 2)
            size = 2.0*max(width, height)

            size_half = np.ceil(size / 2)
            motion = np.random.laplace(0, 1/15, (2,)) * [width, height]
            scale = np.clip(np.random.laplace(1, 1/15), 0.6, 1.4)

            # Calculate regions of interest and maximum padding: max(padding_static, padding_moving)
            sx1 = center[0] - size_half
            sy1 = center[1] - size_half

            sx2 = center[0] + size_half
            sy2 = center[1] + size_half

            # Caclulate padding for static image
            s_pad_left = int(abs(sx1)) if sx1 < 0 else 0
            s_pad_right = int(sx2 - image.shape[1]) if sx2 > image.shape[1] else 0

            s_pad_top = int(abs(sy1)) if sy1 < 0 else 0
            s_pad_bot = int(sy2 - image.shape[0]) if sy2 > image.shape[0] else 0

            mx1 = center[0] - scale*(size_half - motion[0])
            my1 = center[1] - scale*(size_half - motion[1])

            mx2 = center[0] + scale*(size_half + motion[0])
            my2 = center[1] + scale*(size_half + motion[1])

            # Calculate padding for moving image
            m_pad_left = int(abs(mx1)) if mx1 < 0 else 0
            m_pad_right = int(mx2 - image.shape[1]) if mx2 > image.shape[1] else 0

            m_pad_top = int(abs(my1)) if my1 < 0 else 0
            m_pad_bot = int(my2 - image.shape[0]) if my2 > image.shape[0] else 0

            # Calculate joint padding between both images
            pad_bot = max(s_pad_bot, m_pad_bot) + 100
            pad_top = max(s_pad_top, m_pad_top) + 100

            pad_left = max(s_pad_left, m_pad_left) + 100
            pad_right = max(s_pad_right, m_pad_right) + 100

            # Recalculate crop regions after padding
            x1 += pad_left
            x2 += pad_left

            y1 += pad_top
            y2 += pad_top

            center = np.round((np.array([x1, y1]) + np.array([x2, y2])) / 2)

            sx1 = center[0] - size_half
            sy1 = center[1] - size_half

            sx2 = center[0] + size_half
            sy2 = center[1] + size_half

            mx1 = center[0] - scale*(size_half - motion[0])
            my1 = center[1] - scale*(size_half - motion[1])

            mx2 = center[0] + scale*(size_half + motion[0])
            my2 = center[1] + scale*(size_half + motion[1])

            # Pad the actual Image
            image_padded = cv2.copyMakeBorder(image,
                                              pad_top, pad_bot, pad_left, pad_right,
                                              cv2.BORDER_CONSTANT, value=0)

            image_cropped = image_padded[int(sy1):int(sy2), int(sx1):int(sx2)]
            image_resized = cv2.resize(image_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
            siamese_images[0] = image_resized

            image_cropped = image_padded[int(my1):int(my2), int(mx1):int(mx2)]
            image_resized = cv2.resize(image_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
            siamese_images[1] = image_resized

            batch_output_bbox[i, 0] = 224*(x1 - mx1)/(mx2 - mx1)
            batch_output_bbox[i, 1] = 224*(y1 - my1)/(my2 - my1)
            batch_output_bbox[i, 2] = 224*(x2 - mx1)/(mx2 - mx1)
            batch_output_bbox[i, 3] = 224*(y2 - my1)/(my2 - my1)

            # Jointly Normalize Both Images
            siamese_images -= np.mean(siamese_images, axis=(0, 1, 2))
            siamese_images /= np.std(siamese_images, axis=(0, 1, 2)) + np.finfo(np.float32).eps

            batch_input_static[i] = siamese_images[0]
            batch_input_moving[i] = siamese_images[1]

            i += 1

        outputs = []

        if self.regr:
            outputs.append(batch_output_bbox)
        if self.cls:
            outputs.append(batch_output_class)

        return [batch_input_static, batch_input_moving], outputs

    @staticmethod
    # KITTI Format Labels
    def load_kitti_label(image: np.ndarray, label: str):

        label = open(label, 'r').read().strip()

        bboxes = []

        for row in label.split('\n'):
            fields = row.split(' ')

            bbox_class = fields[0]

            bbox_x1 = float(fields[4])
            bbox_y1 = float(fields[5])
            bbox_x2 = float(fields[6])
            bbox_y2 = float(fields[7])

            bbox = ia.BoundingBox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_class)
            bboxes.append(bbox)

        bboi = ia.BoundingBoxesOnImage(bboxes, shape=image.shape)

        return bboi

    @staticmethod
    def create_augmenter(stage: str = "train"):
        if stage == "train":
            return iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
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


def main(batch_size: int = 1):
    import matplotlib.pyplot as plt

    seq = SiameseSequence('train', stage='train')
    for bidx in range(0, len(seq)):

        bins, bouts = seq.__getitem__(bidx)
        static, moving = bins
        bbox, cls = bouts

        for i in range(0, batch_size):

            s = static[i]
            m = moving[i]

            bbox = bbox[i]

            siamese_images = np.array([s, m])

            siamese_images -= np.min(siamese_images)
            siamese_images /= np.max(siamese_images)
            siamese_images *= 255

            siamese_images = siamese_images.astype(np.uint8)

            s = siamese_images[0]
            m = siamese_images[1]

            import cv2
            cv2.rectangle(m, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0))
            plt.imshow(cv2.cvtColor(np.hstack([s, m]), cv2.COLOR_BGR2RGB))
            plt.title('Class: %d' % cls[i])
            plt.show()


if __name__ == '__main__':
    main()
