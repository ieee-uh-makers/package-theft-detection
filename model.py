from keras.models import Model
from keras.layers import *
from keras import layers
import keras.backend as K


def _conv_block(filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1
    filters = int(filters * alpha)

    stack = []

    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')
    stack.append(x)

    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')
    stack.append(x)

    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')
    stack.append(x)

    x = layers.ReLU(6., name='conv1_relu')
    stack.append(x)

    return stack


def _depthwise_conv_block(pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    stack = []

    if strides != (1, 1):
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)
        stack.append(x)

    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)
    stack.append(x)

    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)
    stack.append(x)

    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)
    stack.append(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)
    stack.append(x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)
    stack.append(x)

    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)
    stack.append(x)

    return stack


def build_model(alpha=0.25, depth_multiplier=1, weights: str = 'imagenet', plot: bool = False, cls: bool = False, regr: bool = True):
    siamese_layers = []

    siamese_layers.extend(_conv_block(32, alpha, strides=(2, 2)))
    siamese_layers.extend(_depthwise_conv_block(64, alpha, depth_multiplier, block_id=1))

    siamese_layers.extend(_depthwise_conv_block(128, alpha, depth_multiplier,
                                                strides=(2, 2), block_id=2))
    siamese_layers.extend(_depthwise_conv_block(128, alpha, depth_multiplier, block_id=3))

    siamese_layers.extend(_depthwise_conv_block(256, alpha, depth_multiplier,
                                                strides=(2, 2), block_id=4))
    siamese_layers.extend(_depthwise_conv_block(256, alpha, depth_multiplier, block_id=5))

    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier,
                                                strides=(2, 2), block_id=6))
    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier, block_id=7))

    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier, block_id=8))
    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier, block_id=9))
    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier, block_id=10))
    siamese_layers.extend(_depthwise_conv_block(512, alpha, depth_multiplier, block_id=11))

    siamese_layers.extend(_depthwise_conv_block(1024, alpha, depth_multiplier,
                                                strides=(2, 2), block_id=12))
    siamese_layers.extend(_depthwise_conv_block(1024, alpha, depth_multiplier, block_id=13))

    layer_input_left = Input((224, 224, 3), name='input_left')
    layer_input_right = Input((224, 224, 3), name='input_right')

    x = layer_input_left
    for layer in siamese_layers:
        x = layer(x)
    layer_output_left = x

    x = layer_input_right
    for layer in siamese_layers:
        x = layer(x)
    layer_output_right = x

    conc = Concatenate(name='regr_concat', axis=-1)([layer_output_left, layer_output_right])

    gap = GlobalAveragePooling2D(name='gap')(conc)

    x = gap

    loss_fns = []
    metrics = {}
    outputs = []

    def r2(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    if regr:
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dense(128, activation='relu', name='dense_2')(x)
        x = Dense(128, activation='relu', name='dense_3')(x)
        layer_regr = Dense(4, name='regr')(x)
        outputs.append(layer_regr)
        loss_fns.append('mae')
        metrics['regr'] = r2

    if cls:
        layer_cls = Dense(1, activation='sigmoid', name='cls')(gap)
        outputs.append(layer_cls)
        loss_fns.append('binary_crossentropy')
        metrics['cls'] = 'acc'

    model = Model(inputs=[layer_input_left, layer_input_right],
                  outputs=outputs)

    if cls:
        for layer in model.layers:
            if layer.name != 'cls':
                layer.trainable = False

    model.summary()

    if weights == 'imagenet':
        model.load_weights('weights/mobilenet_2_5_224_tf_no_top.h5', by_name=True)

    if plot:
        from keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)

    return model, loss_fns, metrics


if __name__ == '__main__':
    build_model(cls=True)
