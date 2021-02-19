from tensorflow.keras.layers import *
import models.modelBuilder2D as model_builder_2d
from tensorflow.keras.models import Model
from constants.AI_params import ModelParams


def simpleSimplest(model_params):
    num_filters = 32
    filter_size = 2
    activation = 'relu'
    last_activation = 'relu'
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    number_output_filters = model_params[ModelParams.OUTPUT_SIZE]

    inputs = Input((nn_input_size[0], nn_input_size[1], nn_input_size[2]))

    # 1
    conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(inputs)
    # last
    last_layer = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv1)

    model = Model(inputs=inputs, outputs=[last_layer])
    return model

def simpleCNNBK(model_params, nn_type="2d"):
    num_filters = 32
    filter_size = 2
    activation = 'relu'
    last_activation = 'relu'
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    number_output_filters = model_params[ModelParams.OUTPUT_SIZE]

    if nn_type == "2d":
        inputs = Input((nn_input_size[0], nn_input_size[1], nn_input_size[2]))

        # 1
        conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(inputs)
        batch1 = BatchNormalization()(conv1)
        # 2
        conv2 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(batch1)
        batch2 = BatchNormalization()(conv2)
        # 3
        conv3 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(batch2)
        batch3 = BatchNormalization()(conv3)
        # 4
        conv4 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(batch3)
        batch4 = BatchNormalization()(conv4)
        # last
        last_layer = Conv2D(number_output_filters, (1, 1), activation=last_activation)(batch4)
        # flat_batch4 = Flatten()(batch4)  # Flattens all the neurons
        # units = num_filters * (2 ** (number_output_filters))
        # last_layer = Dense(number_output_filters * nn_input_size[0] * nn_input_size[1], activation='relu')(flat_batch4)

        model = Model(inputs=inputs, outputs=[last_layer])
        return model

    if nn_type == "3d":
        inputs = Input((nn_input_size[0], nn_input_size[1], nn_input_size[2], 1))
        # 1
        conv1 = Conv3D(num_filters, (filter_size, filter_size, filter_size), padding='same', activation=activation)(inputs)
        batch1 = BatchNormalization()(conv1)
        # 2
        conv2 = Conv3D(num_filters, (filter_size, filter_size, filter_size), padding='same', activation=activation)(batch1)
        batch2 = BatchNormalization()(conv2)
        # 3
        conv3 = Conv3D(num_filters, (filter_size, filter_size, filter_size), padding='same', activation=activation)(batch2)
        batch3 = BatchNormalization()(conv3)
        # 4
        conv4 = Conv3D(num_filters, (filter_size, filter_size, filter_size), padding='same', activation=activation)(batch3)
        batch4 = BatchNormalization()(conv4)
        # last
        last_layer = Conv3D(number_output_filters, (1, 1, 1), activation=last_activation)(batch4)
        # flat_batch4 = Flatten()(batch4)  # Flattens all the neurons
        # units = num_filters * (2 ** (number_output_filters))
        # last_layer = Dense(number_output_filters * nn_input_size[0] * nn_input_size[1], activation='relu')(flat_batch4)

        model = Model(inputs=inputs, outputs=[last_layer])
        return model

def simpleCNN(model_params, nn_type="2d", hid_lay=2, out_lay=2):
    num_filters = 32
    filter_size = 3
    activation = 'relu'
    last_activation = 'relu'
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    number_output_filters = model_params[ModelParams.OUTPUT_SIZE]

    if nn_type == "2d":
        inputs = Input((nn_input_size[0], nn_input_size[1], nn_input_size[2]))
        for layer in range(hid_lay):
            if layer == 0:
                cnn_layer, _ = model_builder_2d.multiple_conv_lay_2d(inputs, num_filters, filter_size, make_pool=False, batch_norm=True,
                                                                    activation=LeakyReLU(alpha=0.2), dropout=False, tot_layers=1)
            else:
                cnn_layer, _ = model_builder_2d.multiple_conv_lay_2d(cnn_layer, num_filters*2, filter_size, make_pool=False, batch_norm=True,
                                                                    activation=LeakyReLU(alpha=0.2), dropout=False, tot_layers=1)

        for dense_lay in range(out_lay-1):
            cnn_layer = Conv2D(number_output_filters, (1, 1), activation=LeakyReLU(alpha=0.2))(cnn_layer)

        last_layer = Conv2D(number_output_filters, (1, 1), activation=LeakyReLU(alpha=0.2))(cnn_layer)
        model = Model(inputs=inputs, outputs=[last_layer])
        return model

def simpleCNNDenseWrong(model_params, nn_type="2d"):
    nn_input_size = model_params[ModelParams.INPUT_SIZE]
    number_output_filters = model_params[ModelParams.OUTPUT_SIZE]

    activation = 'relu'
    last_activation = 'relu'

    if nn_type == "2d":
        inputs = Input((nn_input_size[0], nn_input_size[1], nn_input_size[2]))

        flat_input = Flatten()(inputs)  # Flattens all the neurons
        d1 = Dense(1024, activation=activation)(flat_input)
        d2 = Dense(512, activation=activation)(d1)
        d3 = Dense((nn_input_size[0] * nn_input_size[1] * number_output_filters), activation=activation)(d2)
        last_layer = Reshape((nn_input_size[0], nn_input_size[1], number_output_filters))(d3)
        model = Model(inputs=inputs, outputs=[last_layer])
        return model

