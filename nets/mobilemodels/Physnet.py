import tensorflow as tf
import numpy as np
from typeguard import typechecked
from typing import Union, Callable, Iterable
import h5py
import json
import keras_utils as conv_utils
import os
from tensorflow.keras.utils import Sequence

tf.config.run_functions_eagerly(True)

os.environ["CUDA_VISIBLE_DEVICES"]="3"
print(tf.__version__)
print("here")

IMG_SIZE = 36
NUM_FEATURES =9 * 9 * 64 # Deepphy base line model output feature
NUM_CLASSES = 1          # Deepphy output : delta bvp
# Base Model
# DeepPhys Model
# input ( None, 36, 36, 6 )
#     MotionModel(_in[0])  ──┬──┬── [F]
#                            │  │
# AppearanceModel(_in[1])  ──┴──┘
class AdaptivePooling3D(tf.keras.layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.
    This class only exists for code reuse. It will never be an exposed API.
    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 3 integers specifying (pooled_dim1, pooled_dim2, pooled_dim3).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 3, "output_size")
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        d_bins = self.output_size[2]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = self.reduce_function(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = self.reduce_function(split_depth, axis=[3, 5, 7])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                    input_shape[4],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                ]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    print(predictions,targets)
    rst = 0
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = tf.cast(predictions, tf.float32)
    predictions = (predictions - tf.math.reduce_mean(predictions)) / tf.math.reduce_std(predictions)
    targets = (targets - tf.math.reduce_mean(targets)) / tf.math.reduce_std(targets)

    for i in range(1):
        sum_x = tf.math.reduce_sum(predictions[i])  # x
        sum_y = tf.math.reduce_sum(targets[i])  # y
        sum_xy = tf.math.reduce_sum(predictions[i] * targets[i])  # xy
        sum_x2 = tf.math.reduce_sum(tf.math.pow(predictions[i], 2))  # x^2
        sum_y2 = tf.math.reduce_sum(tf.math.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            tf.math.sqrt((N * sum_x2 - tf.math.pow(sum_x, 2)) * (N * sum_y2 - tf.math.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / 1
    return rst
@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling3D(AdaptivePooling3D):
    """Average Pooling with adaptive kernel size.
    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.
    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)
class Deepphys(tf.keras.Model): # base model
    def __init__(self):
        super(Deepphys,self).__init__()
        self.a_1 = tf.keras.layers.ZeroPadding2D(1)
        self.a_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,name='a_2')
        self.a_3 = tf.keras.layers.BatchNormalization()

        self.a_4 = tf.keras.layers.ZeroPadding2D(1)
        self.a_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=3,name='a_5')
        self.a_6 = tf.keras.layers.BatchNormalization()

        self.a_7 = tf.keras.layers.AveragePooling2D(pool_size =2,strides=2)

        self.a_8 = tf.keras.layers.ZeroPadding2D(1)
        self.a_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,name='a_7')
        self.a_10 = tf.keras.layers.BatchNormalization()

        self.a_11 = tf.keras.layers.ZeroPadding2D(1)
        self.a_12 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,name='a_12')
        self.a_13 = tf.keras.layers.BatchNormalization()

        self.att_conv_1 = tf.keras.layers.Conv2D(filters=1,kernel_size=1,activation='sigmoid',name='att_1')
        self.att_conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1,activation='sigmoid',name='att_2')

        self.m_1 = tf.keras.layers.ZeroPadding2D(1)
        self.m_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='valid',name='m_2')
        self.m_3 = tf.keras.layers.BatchNormalization()

        self.m_4 = tf.keras.layers.ZeroPadding2D(1)
        self.m_5 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,name='m_5')
        self.m_6 = tf.keras.layers.BatchNormalization()

        self.m_7 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.m_8 = tf.keras.layers.ZeroPadding2D(1)
        self.m_9 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,name='m_9')
        self.m_10 = tf.keras.layers.BatchNormalization()

        self.m_11 = tf.keras.layers.ZeroPadding2D(1)
        self.m_12 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,name='m_12')
        self.m_13 = tf.keras.layers.BatchNormalization()

        self.m_14 = tf.keras.layers.AveragePooling2D(pool_size=2,strides=2)

        self.f_1 = tf.keras.layers.Flatten()
        self.f_2 = tf.keras.layers.Dense(256)
        self.f_3 = tf.keras.layers.Dense(1)


    @tf.function(input_signature=[
        tf.TensorSpec([None,36,36,6], tf.float32),
    ])
    def call(self,inputs):
        _in = tf.split(inputs,2,axis=3)
        A = self.a_1(_in[1])
        A = self.a_2(A)
        # A = self.a_3(A)
        A = tf.keras.activations.tanh(A)

        A = self.a_4(A)
        A = self.a_5(A)
        # A = self.a_6(A)
        A = tf.keras.activations.tanh(A)

        M1 = self.att_conv_1(A)
        # M1 = tf.keras.activations.sigmoid(M1)
        # B,_,H,W = tf.shape(M1)
        norm = tf.abs(M1)
        norm = tf.math.reduce_sum(norm)
        norm = 2 * norm
        norm = tf.reshape(norm,[1,1,1,1])
        M1 = tf.divide(M1,norm)

        A = self.a_7(A)
        A = self.a_8(A)
        A = self.a_9(A)
        # A = self.a_10(A)
        A = tf.keras.activations.tanh(A)

        A = self.a_11(A)
        A = self.a_12(A)
        # A = self.a_13(A)
        A = tf.keras.activations.tanh(A)

        M2 = self.att_conv_2(A)
        # M2 = tf.keras.activations.sigmoid(M2)
        # B, _, H, W = tf.shape(M2)
        norm = tf.abs(M2)
        norm = tf.math.reduce_sum(norm)
        norm = 2 * norm
        norm = tf.reshape(norm, [1, 1, 1, 1])
        M2 = tf.divide(M2, norm)

        M = self.m_1(_in[0])
        M = self.m_2(M)
        # M = self.m_3(M)
        M = tf.keras.activations.tanh(M)

        M = self.m_4(M)
        M = self.m_5(M)
        # M = self.m_6(M)

        ones_1 = tf.ones([1,36,36,36])
        mat_1 = ones_1 @ M1 #tf.matmul(ones_1,M1)
        g1 = mat_1*M
        M = tf.keras.activations.tanh(g1)

        M = self.m_7(M)

        M = self.m_8(M)
        M = self.m_9(M)
        # M = self.m_10(M)
        M = tf.keras.activations.tanh(M)

        M = self.m_11(M)
        M = self.m_12(M)
        # M = self.m_13(M)

        ones_2 = tf.ones([1,18,18,18])
        mat_2 = ones_2 @ M
        g2 = mat_2 * M
        M = tf.keras.activations.tanh(g2)

        M = tf.keras.activations.tanh(M)

        M = self.m_14(M)

        F = self.f_1(M)
        F = self.f_2(F)
        F = self.f_3(F)

        return F
# TRF Model
# BaseModel(bottleneck) ───── [output]

class TransferLearningModel(tf.Module):

    def __init__(self, learning_rate = 0.001):

        self.base = Deepphys()
        self.transfer_layer = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(1)
        ])
        # self.loss_fn = tf.keras.losses.mean_squared_error()
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.transfer_layer.compile(
            optimizer='adam',
            loss=tf.keras.losses.mean_squared_error)

    @tf.function(input_signature=[
        tf.TensorSpec([None,IMG_SIZE,IMG_SIZE,6],tf.float32),
        ])
    def load(self, feature):
        bottleneck = tf.reshape(self.base(feature, training=False),(-1,NUM_FEATURES))
        return {'bottleneck': bottleneck}

    @tf.function(input_signature=[
        tf.TensorSpec([None,NUM_FEATURES], tf.float32),
        tf.TensorSpec([NUM_CLASSES,], tf.float32)
    ])
    def train(self, bottleneck, label):
        with tf.GradientTape() as tape:
            predictions = self.transfer_layer(bottleneck)
            loss = self.transfer_layer.loss(predictions,label)
        gradients = tape .gradient(loss, self.transfer_layer.trainable_variables)
        self.transfer_layer.optimizer.apply_gradients(
            zip(gradients, self.transfer_layer.trainable_variables))
        result = {"loss": loss}
        for grad in gradients:
            result[grad.name] = grad
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None,IMG_SIZE,IMG_SIZE,6],tf.float32)
    ])
    def infer(self,x):
        output = self.transfer_layer(self.base(x))
        return {
            "output": output
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.base.weights]
        tensors_to_save = [weight.read_value() for weight in self.base.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.base.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors
class physnet(tf.keras.Model):
    def __init__(self):
        super(physnet, self).__init__()

        self.model = tf.keras.Sequential([
        # EncoderBlock
        # ConvBlock1
        tf.keras.layers.ZeroPadding3D(padding=(0,2,2),input_shape=(32,128,128,3)),
        tf.keras.layers.Conv3D(filters=16,kernel_size=(1,5,5),strides=(1,1,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        #MaxPool3d1r
        # tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2)),
        tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 2, 2), strides=(1, 2, 2)),
        # #
        # # ConvBlock2
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # ConvBlock3
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        # # MaxPool3d2
        tf.keras.layers.Conv3D(filters=64, kernel_size=(2, 2, 2), strides=(2, 2, 2)),
        # tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
        #
        # ConvBlock4
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # ConvBlock5
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        #
        # # MaxPool3d3
        # tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(2, 2, 2), strides=(2, 2, 2)),
        #
        # # ConvBlock6
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # ConvBlock7
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        #
        # # MaxPool3d4
        # tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 2, 2), strides=(1, 2, 2)),
        # #
        # # ConvBlock8
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # ConvBlock9
        tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
        tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        # # #
        # # # DecoderBlock
        # # #
        # DeConvBlock1
        tf.keras.layers.Convolution3DTranspose(filters=64,kernel_size=(4,1,1),strides=(2,1,1),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU(),

        # DeConvBlock2
        tf.keras.layers.Convolution3DTranspose(filters=64, kernel_size=(4, 1, 1), strides=(2, 1, 1),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ELU(),

        #
        # AdaptivePooling
        #
        AdaptivePooling3D(tf.reduce_max,(32,1,1)),
        # #
        #Conv3D
        #
        # tf.keras.layers.Conv3D(filters=32,kernel_size=(8,8,1)),

        tf.keras.layers.Conv3D(1,kernel_size=(1,64,64),strides=(1,1,1),padding='same'),
        tf.keras.layers.Reshape((-1,))
        ])
    @tf.function(input_signature=[
        tf.TensorSpec([None,32,128,128,3], tf.float32),

    ])
    def call(self,inputs):
        return self.model(inputs)
class Model(tf.Module):
    def __init__(self):
        self.model = Deepphys()
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.mean_squared_error)


    @tf.function(input_signature=[
        tf.TensorSpec([None, 36,36,6], tf.float32),
        tf.TensorSpec([None, 1 ], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss(predictions, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss":loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None,36,36,6], tf.float32),
    ])
    def infer(self, x):

        output = self.model(x)
        return {
            "output" : output
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors
    @tf.function(input_signature=[
        tf.TensorSpec([None,36,36,3],dtype = tf.float32),
        tf.TensorSpec([None,36,36,3],dtype = tf.float32),
    ])
    def preprocessing(self,img1,img2):
        img1 = tf.cast(img1,tf.float32)
        img1 = tf.divide(img1,255.0)

        img2 = tf.cast(img2, tf.float32)
        img2 = tf.divide(img2, 255.0)

        motion_input = tf.divide(tf.subtract(img2,img1),tf.add(img2,img1))
        norm_input = tf.divide(img1,tf.math.reduce_std(img1))
        result = tf.concat([motion_input,norm_input],axis=-1)
        return result
class physnetModel(tf.Module):

    def __init__(self):
        self.model = physnet()
        self.model.compile( optimizer='adam', loss=neg_Pearson_Loss )

    @tf.function(input_signature=[
        tf.TensorSpec([None, 32, 128, 128, 3], tf.float32),
        tf.TensorSpec([None, 1], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        result = {"loss":loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None,32,128,128,3], tf.float32),
    ])
    def infer(self, x):
        output = self.model(x)
        return {
            "output" : output
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

with open('../../params.json') as f:
    jsonObject = json.load(f)
    params = jsonObject.get("params")
    model_params = jsonObject.get("model_params")
    save_root_path = params["save_root_path"]
    model_name = model_params["name"]
    dataset_name = params["dataset_name"]
    option="train"

hpy_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_" + option + ".hdf5", "r")
video_data = []
label_data = []
for key in hpy_file.keys():
     video_data.extend(hpy_file[key]['preprocessed_video'])
     label_data.extend(hpy_file[key]['preprocessed_label'])
hpy_file.close()

# train_loader = DataLoader(video_data,label_data,1)

# train_loader = DataLoader(video_data[:(int)(video_data.__len__()*0.8)],label_data[:(int)(label_data.__len__()*0.8)],1)
# valid_loader = DataLoader(video_data[(int)(video_data.__len__()*0.8):],label_data[(int)(label_data.__len__()*0.8):],1)
# test_loader = DataLoader(video_data,label_data,32)

NUM_EPOCHS = 10
BATCH_SIZE = 1
epochs = np.arange(1, NUM_EPOCHS + 1, 1)
losses = np.zeros([NUM_EPOCHS])
m = Model()
# m.model.build(input_shape=(1,36,36,6))
# m.model.summary()
video_data = np.asarray(video_data)
label_data = np.asarray(label_data)
train_ds = tf.data.Dataset.from_tensor_slices((video_data[:2],label_data[:2]))
train_ds = train_ds.batch(BATCH_SIZE)

# trian test gpu
for i in range(NUM_EPOCHS):
  for x,y in train_ds:
    x = tf.cast(x,tf.float32)
    result = m.train(x, y)
  losses[i] = result['loss']
  if (i + 1) % 1 == 0:
      # print(result['loss'])
   print(f"Finished {i+1} epochs")
   print(f"  loss: {losses[i]:.3f}")

# tflite save path
m.save('/tmp/trained_model')

SAVED_MODEL_DIR = "./tmp"
tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures={
        'train':
            m.train.get_concrete_function(),
        'infer':
            m.infer.get_concrete_function(),
        'save':
            m.save.get_concrete_function(),
        'restore':
            m.restore.get_concrete_function(),
        'preprocessing':
            m.preprocessing.get_concrete_function(),
    })

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
f = open('test.tflite','wb')
f.write(tflite_model)
f.close()

# get model from tflite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
# select model [ infer/ train/ save/ restore ]
preprocess = interpreter.get_signature_runner("preprocessing")
import PIL.Image as pilimg
import numpy as np
im1 = pilimg.open("/media/hdd1/UBFC/subject1/frames/frame0001.bmp")
im2 = pilimg.open("/media/hdd1/UBFC/subject1/frames/frame0002.bmp")
im1 = np.array(im1)
im2 = np.array(im2)
im1 = tf.cast(im1,tf.float32)
im2 = tf.cast(im2,tf.float32)
im1 = tf.expand_dims(im1,axis=0)
im2 = tf.expand_dims(im2,axis=0)
b = preprocess(img1=im1,img2=im2)
# infer = interpreter.get_signature_runner("infer")
# for x, _ in train_ds:
#     x = tf.cast(x, tf.float32)
#     logits_original = m.infer(x=x)['output'][0]
#     logits_lite = infer(x=x)['output'][0]
#
#     # print(logits_original, logits_lite)
#
#
# train = interpreter.get_signature_runner("train")
#
# for i in range(NUM_EPOCHS):
#   for x,y in train_ds:
#     x = tf.cast(x,tf.float32)
#     y = tf.cast(y,tf.float32)
#     result = train(x=x,y=y)
#     print(result)