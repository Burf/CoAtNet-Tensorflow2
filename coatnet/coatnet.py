import numpy as np
import tensorflow as tf

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, emb_dim = 768, n_head = 12, out_dim = None, relative_window_size = None, dropout_rate = 0., kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01), **kwargs):
        #ScaledDotProductAttention
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_head = n_head
        if emb_dim % n_head != 0:
            raise ValueError("Shoud be embedding dimension % number of heads = 0.")
        if out_dim is None:
            out_dim = self.emb_dim
        self.out_dim = out_dim
        if relative_window_size is not None and np.ndim(relative_window_size) == 0:
            relative_window_size = [relative_window_size, relative_window_size]
        self.relative_window_size = relative_window_size
        self.projection_dim = emb_dim // n_head
        self.dropout_rate = dropout_rate
        self.query = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.key = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.value = tf.keras.layers.Dense(emb_dim, kernel_initializer = kernel_initializer)
        self.combine = tf.keras.layers.Dense(out_dim, kernel_initializer = kernel_initializer)
        
    def build(self, input_shape):
        if self.relative_window_size is not None:
            self.relative_position_bias_table = self.add_weight("relative_position_bias_table", shape = [((2 * self.relative_window_size[0]) - 1) * ((2 * self.relative_window_size[1]) - 1), self.n_head], trainable = self.trainable)
            coords_h = np.arange(self.relative_window_size[0])
            coords_w = np.arange(self.relative_window_size[1])
            coords = np.stack(np.meshgrid(coords_h, coords_w, indexing = "ij")) #2, Wh, Ww
            coords = np.reshape(coords, [2, -1])
            relative_coords = np.expand_dims(coords, axis = -1) - np.expand_dims(coords, axis = -2) #2, Wh * Ww, Wh * Ww
            relative_coords = np.transpose(relative_coords, [1, 2, 0]) #Wh * Ww, Wh * Ww, 2
            relative_coords[:, :, 0] += self.relative_window_size[0] - 1 #shift to start from 0
            relative_coords[:, :, 1] += self.relative_window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.relative_window_size[1] - 1
            relative_position_index = np.sum(relative_coords, -1)
            self.relative_position_index = tf.Variable(tf.convert_to_tensor(relative_position_index), trainable = False, name= "relative_position_index")
        
    def attention(self, query, key, value, relative_position_bias = None):
        score = tf.matmul(query, key, transpose_b = True)
        n_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(n_key)
        if relative_position_bias is not None:
            scaled_score = scaled_score + relative_position_bias
        weight = tf.nn.softmax(scaled_score, axis = -1)
        if 0 < self.dropout_rate:
            weight = tf.nn.dropout(weight, self.dropout_rate)
        out = tf.matmul(weight, value)
        return out
    
    def separate_head(self, x):
        out = tf.keras.layers.Reshape([-1, self.n_head, self.projection_dim])(x)
        out = tf.keras.layers.Permute([2, 1, 3])(out)
        return out
    
    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        query = self.separate_head(query)
        key = self.separate_head(key)
        value = self.separate_head(value)
        
        relative_position_bias = None
        if self.relative_window_size is not None:
            relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
            relative_position_bias = tf.reshape(relative_position_bias, [self.relative_window_size[0] * self.relative_window_size[1], self.relative_window_size[0] * self.relative_window_size[1], -1]) #Wh * Ww,Wh * Ww, nH
            relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1]) #nH, Wh * Ww, Wh * Ww
            relative_position_bias = tf.expand_dims(relative_position_bias, axis = 0)
        attention = self.attention(query, key, value, relative_position_bias)
        attention = tf.keras.layers.Permute([2, 1, 3])(attention)
        attention = tf.keras.layers.Reshape([-1, self.emb_dim])(attention)
        
        out = self.combine(attention)
        return out
        
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config["emb_dim"] = self.emb_dim
        config["n_head"] = self.n_head
        config["out_dim"] = self.out_dim
        config["relative_window_size"] = self.relative_window_size
        config["projection_dim"] = self.projection_dim
        config["dropout_rate"] = self.dropout_rate
        return config

class MBConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides = 1, expand_ratio = 1, se_ratio = 4, momentum = 0.9, epsilon = 0.01, convolution = tf.keras.layers.Conv2D, activation = tf.nn.swish, kernel_initializer = "he_normal", **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.momentum = momentum
        self.epsilon = epsilon
        self.convolution = convolution
        self.activation = activation
        
    def build(self, input_shape):
        self.layers = []
        self.downsample = []
        if self.expand_ratio != 1:
            conv = self.convolution(input_shape[-1] * self.expand_ratio, 1, use_bias = False)
            norm = tf.keras.layers.BatchNormalization(momentum = self.momentum, epsilon = self.epsilon)
            act = tf.keras.layers.Activation(self.activation)
            input_shape = input_shape[:-1] + (input_shape[-1] * self.expand_ratio,)
            self.layers += [conv, norm, act]
        
        #Depthwise Convolution
        conv = self.convolution(input_shape[-1], self.kernel_size, strides = self.strides, groups = input_shape[-1], padding = "same", use_bias = False)
        norm = tf.keras.layers.BatchNormalization(momentum = self.momentum, epsilon = self.epsilon)
        act = tf.keras.layers.Activation(self.activation)
        self.layers += [conv, norm, act]
        
        #Squeeze and Excitation layer, if desired
        axis = list(range(1, len(input_shape) - 1))
        gap = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis = axis, keepdims = True))
        squeeze = self.convolution(max(1, int(input_shape[-1] / self.se_ratio)), 1, use_bias = True)
        act = tf.keras.layers.Activation(self.activation)
        excitation = self.convolution(input_shape[-1], 1, use_bias = True)
        se = lambda x: x * tf.nn.sigmoid(excitation(act(squeeze(gap(x)))))
        self.layers += [se]
        
        #Output Phase
        conv = self.convolution(self.filters, 1, use_bias = False)
        norm = tf.keras.layers.BatchNormalization(momentum = self.momentum, epsilon = self.epsilon)
        self.layers += [conv, norm]
        
    def call(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            
        #Skip Connection
        if self.strides == 1 and tf.keras.backend.int_shape(x)[-1] == tf.keras.backend.int_shape(out)[-1]:
            out = out + x
        return out
        
    def get_config(self):
        config = super(MBConv, self).get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["expand_ratio"] = self.expand_ratio
        config["se_ratio"] = self.se_ratio
        config["momentum"] = self.momentum
        config["epsilon"] = self.epsilon
        config["convolution"] = self.convolution
        config["activation"] = self.activation
        return config
    
def coatnet(x, n_class = 1000, include_top = True, n_depth = [2, 2, 6, 14, 2], n_feature = [64, 96, 192, 384, 768], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu, name = ""):
    #block : S > Stem, C > MBConv, T > Transformer
    if 0 < len(name):
        name += "_"
        
    out = x
    for i, (_n_depth, _n_feature, _block) in enumerate(zip(n_depth, n_feature, block)):
        for j in range(_n_depth):
            stride_size = 1 if j != 0 else 2
            residual = out
            if _block.upper() == "C":# i == 0:
                out = tf.keras.layers.Conv2D(_n_feature, 1 if i != 0 else 3, strides = stride_size, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "{0}stage{1}_conv{2}".format(name, i, j + 1))(out)
                out = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "{0}stage{1}_norm{2}".format(name, i, j + 1))(out)
                out = tf.keras.layers.Activation(activation, name = "{0}stage{1}_act{2}".format(name, i, j + 1))(out)
            elif _block.upper() == "M":
                out = tf.keras.layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "{0}stage{1}_pre_norm{2}".format(name, i, j + 1))(out)
                out = MBConv(_n_feature, 3, strides = stride_size, expand_ratio = expand_ratio, se_ratio = se_ratio, momentum = 0.9, epsilon = 1e-5, activation = activation, name = "{0}stage{1}_mbconv{2}".format(name, i, j + 1))(out)
                if stride_size == 2:
                    residual = tf.keras.layers.MaxPool2D(pool_size = 3, strides = stride_size, padding = "same", name = "{0}stage{1}_residual_pooling".format(name, i))(residual)
                    residual = tf.keras.layers.Conv2D(_n_feature, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "{0}stage{1}_residual_conv".format(name, i))(residual)
                    out = tf.keras.layers.Add(name = "{0}stage{1}_residual_add".format(name, i))([out, residual])
            elif _block.upper() == "T":
                out = tf.keras.layers.LayerNormalization(epsilon = 1e-5, name = "{0}stage{1}_pre_norm{2}".format(name, i, j + 1))(out)
                shape = tf.keras.backend.int_shape(out)[1:3]
                if stride_size == 2:
                    shape = np.divide(shape, stride_size).astype(int)
                    out = tf.keras.layers.MaxPool2D(pool_size = 3, strides = stride_size, padding = "same", name = "{0}stage{1}_pooling".format(name, i))(out)
                    residual = tf.keras.layers.MaxPool2D(pool_size = 3, strides = stride_size, padding = "same", name = "{0}stage{1}_residual_pooling".format(name, i))(residual)
                    residual = tf.keras.layers.Conv2D(_n_feature, 1, padding = "same", use_bias = False, kernel_initializer = "he_normal", name = "{0}stage{1}_residual_conv".format(name, i))(residual)
                out = tf.keras.layers.Reshape([-1, tf.keras.backend.int_shape(out)[-1]], name = "{0}stage{1}_pre_reshape{2}".format(name, i, j + 1))(out)
                out = MultiHeadSelfAttention(emb_dim = 32 * 8, n_head = 8, out_dim = _n_feature, relative_window_size = shape, name = "{0}stage{1}_attention{2}".format(name, i, j + 1))(out)
                out = tf.keras.layers.Reshape([*shape, tf.keras.backend.int_shape(out)[-1]], name = "{0}stage{1}_post_reshape{2}".format(name, i, j + 1))(out)
                out = tf.keras.layers.Add(name = "{0}stage{1}_residual_add{2}".format(name, i, j + 1))([out, residual])
                
                out = tf.keras.layers.LayerNormalization(epsilon = 1e-5, name = "{0}stage{1}_ffn{2}_pre_norm".format(name, i, j + 1))(out)
                out = tf.keras.layers.Dense(_n_feature, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01), name = "{0}stage{1}_ffn{2}_dense1".format(name, i, j + 1))(out)
                out = tf.keras.layers.Activation(activation, name = "{0}stage{1}_ffn{2}_act".format(name, i, j + 1))(out)
                out = tf.keras.layers.Dense(_n_feature, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01), name = "{0}stage{1}_ffn{2}_dense2".format(name, i, j + 1))(out)

    if include_top:
        out = tf.keras.layers.GlobalAveragePooling2D(name = "{0}gap".format(name))(out)
        if 0 < dropout_rate:
            out = tf.keras.layers.Dropout(dropout_rate, name = "{0}dropout".format(name))(out)
        out = tf.keras.layers.Dense(n_class, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01), name = "{0}logits".format(name))(out)
    return out

def coatnet0(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 3, 5, 2], n_feature = [64, 96, 192, 384, 768], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model

def coatnet1(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 6, 14, 2], n_feature = [64, 96, 192, 384, 768], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model

def coatnet2(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 6, 14, 2], n_feature = [128, 128, 256, 512, 1024], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model

def coatnet3(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 6, 14, 2], n_feature = [192, 192, 384, 768, 1536], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model

def coatnet4(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 12, 28, 2], n_feature = [192, 192, 384, 768, 1536], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model

def coatnet5(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = None):
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape = input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    out = coatnet(img_input, classes, include_top, n_depth = [2, 2, 12, 28, 2], n_feature = [192, 256, 512, 1280, 2048], block = ["C", "M", "M", "T", "T"], expand_ratio = 4, se_ratio = 4, dropout_rate = 0., activation = tf.keras.activations.gelu)
    model = tf.keras.Model(img_input, out)
    
    if weights is not None:
        model.load_weights(weights)
    return model