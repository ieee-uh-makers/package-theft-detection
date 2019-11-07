from keras.layers import *
from keras.models import Model
from keras.applications import MobileNet
from typing import Optional
from keras.regularizers import l2


def build_model(stage: str = "train",
                timesteps: int = 10,
                input_shape: int = 225,
                output_dims: int = 256,
                weights: Optional[str] = None):

    layer_input_train_img = Input((timesteps, input_shape, input_shape, 3), name='input_img_sequence')

    lstm_layers = [LSTM(output_dims, name='lstm'),
                   Dense(1, activation='sigmoid', name='cls', kernel_regularizer=l2(0.001))]

    if stage == "train" or stage == "inf_cnn":
        model_mnet = MobileNet(alpha=0.25, weights='imagenet', input_shape=(input_shape, input_shape, 3),
                               include_top=False)

        # Training Network
        cnn_layers = [layer for layer in model_mnet.layers[2:]]
        for layer in cnn_layers:
            layer.trainable = False
        cnn_layers.append(GlobalMaxPooling2D(name='gmp'))
        cnn_layers.append(RepeatVector(1, name='repeat'))

        if stage == "train":

            cnn_features = []
            for i in range(0, timesteps):

                x = layer_input_train_img
                x = Lambda(lambda s: s[:, i, :, :], name='input_image_timestep_%d' % (i + 1))(x)

                for layer in cnn_layers:
                    x = layer(x)

                cnn_features.append(x)

            x = Concatenate(name='concat_features', axis=1)(cnn_features)

            for layer in lstm_layers:
                x = layer(x)

            model_train = Model(inputs=layer_input_train_img, outputs=x)
            if weights is not None:
                model_train.load_weights('weights/mobilenet_2_5_224_tf_no_top.h5', by_name=True)
                model_train.load_weights(weights, by_name=True)

            metrics = {'cls': 'accuracy'}
            loss_fns = ['binary_crossentropy']

            return model_train, loss_fns, metrics
        elif stage == "inf_cnn":
            # Convolution Inference Network
            layer_input_inf_img = Input((input_shape, input_shape, input_shape), name='input_image')

            x = layer_input_inf_img
            for layer in cnn_layers:
                x = layer(x)

            model_inf_conv = Model(inputs=layer_input_inf_img, outputs=x)
            if weights is not None:
                model_inf_conv.load_weights('weights/mobilenet_2_5_224_tf_no_top.h5', by_name=True)
                model_inf_conv.load_weights(weights, by_name=True)

            return model_inf_conv, None, None

    elif stage == "inf_lstm":
        # LSTM Inference Network
        layer_input_features = Input((timesteps, output_dims), name='input_features')
        x = layer_input_features
        for layer in lstm_layers:
            x = layer(x)

        model_inf_lstm = Model(inputs=layer_input_features, outputs=x)
        if weights is not None:
            model_inf_lstm.load_weights(weights, by_name=True)
        return model_inf_lstm, None, None


if __name__ == '__main__':
    model, _, _ = build_model()
    model.summary()


