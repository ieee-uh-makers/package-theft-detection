import os
import plac
import time

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping

from model import build_model
from adabound import AdaBound
from rectified_adam import RectifiedAdam

from generator import SiameseSequence


@plac.annotations(
    session=('Name of the training session', 'option', 'S', str),
    stage=('Training stage (regr or cls)', 'option', 's', str),
    batch_size=('The training batch size', 'option', 'B', int),
    epochs=('Number of epochs to train', 'option', 'E', int),
    train_path=(
            'Path to the train folder which contains both an images and labels folder with KITTI labels',
            'option', 'T', str),
    val_path=(
            'Path to the validation folder which contains both an images and labels folder with KITTI labels',
            'option', 'V', str),
    weights=('Weights file to start with', 'option', 'W', str),
    workers=('Number of fit_generator workers', 'option', 'w', int)
)
def main(session: str = time.strftime("%Y-%m-%d_%H-%M-%S"),
         stage: str = 'regr',
         batch_size: int = 24,
         epochs: int = 384,
         train_path: str = 'train',
         val_path: str = 'val',
         weights=None,
         workers: int = 12):

    regr = True
    cls = True if stage == 'cls' else False

    model, loss_fns, metrics = build_model(regr=regr, cls=cls)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    train_seq = SiameseSequence(train_path, stage="train", batch_size=batch_size, cls=cls, regr=regr)
    val_seq = SiameseSequence(val_path, stage="val", batch_size=batch_size, cls=cls, regr=regr)

    callbacks = []

    model.compile(optimizer=RectifiedAdam(lr=0.0001, clipnorm=5.0), loss=loss_fns, metrics=metrics)

    try:
        os.mkdir('weights')
    except FileExistsError:
        pass

    if stage == 'regr':
        filepath = "weights/%s_epoch {epoch:02d}_r2 {val_r2:.4f}.h5" % session
        checkpoint = ModelCheckpoint(filepath, monitor='val_r2', verbose=1, save_best_only=True, mode='max')
    elif stage == 'cls':
        filepath = "weights/%s_epoch {epoch:02d}_r2 {val_cls_acc:.4f}.h5" % session
        checkpoint = ModelCheckpoint(filepath, monitor='val_cls_acc', verbose=1, save_best_only=True, mode='max')

    callbacks.append(checkpoint)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    callbacks.append(early_stop)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1, mode='min')
    callbacks.append(reduce_lr)

    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    tensorboard = TensorBoard(log_dir='logs/%s' % session)
    callbacks.append(tensorboard)

    model.fit_generator(train_seq,
                        validation_data=val_seq,
                        epochs=epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=workers,
                        shuffle=True)


if __name__ == '__main__':
    plac.call(main)
