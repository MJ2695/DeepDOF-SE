import tensorflow as tf
import tensorflow_addons as tfa

IMG_CHANNELS = 3
NUM_GEN_FILTERS = 32

def resnet_block(filters=NUM_GEN_FILTERS*4, size=3):
    inputs = tf.keras.layers.Input(shape=[None, None, filters])
    
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    res_layer1 = tf.keras.layers.Conv2D(filters, size, strides=(1,1), padding='valid',
                                        kernel_initializer = tf.random_normal_initializer(0., .02))
    res_instancenorm1 = tfa.layers.InstanceNormalization(axis = 3,
                                                         epsilon = 1e-5,
                                                         center = True,
                                                         scale = True,
                                                         beta_initializer = 'random_uniform',
                                                         gamma_initializer = 'random_uniform')
    
    res_layer2 = tf.keras.layers.Conv2D(filters, size, strides=(1,1), padding='valid',
                                        kernel_initializer = tf.random_normal_initializer(0., .02))
    res_instancenorm2 = tfa.layers.InstanceNormalization(axis = 3,
                                                         epsilon = 1e-5,
                                                         center = True,
                                                         scale = True,
                                                         beta_initializer = 'random_uniform',
                                                         gamma_initializer = 'random_uniform')
    
    pad_input = tf.pad(inputs, paddings, "REFLECT")
    o_c1 = res_layer1(pad_input)
    o_c1 = res_instancenorm1(o_c1)
    pad_o_c1 = tf.pad(o_c1, paddings, "REFLECT")
    o_c2 = res_layer2(pad_o_c1)
    o_c2 = res_instancenorm2(o_c2)

    outputs = tf.nn.relu(o_c2+inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def generator_downsample():
    initializer = tf.random_normal_initializer(0.,0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(NUM_GEN_FILTERS, 7, strides=(1,1), padding='valid',
                                     kernel_initializer = initializer))
    model.add(tfa.layers.InstanceNormalization(axis = 3,
                                               epsilon = 1e-5,
                                               center = True,
                                               scale = True,
                                               beta_initializer = 'random_uniform',
                                               gamma_initializer = 'random_uniform'))
    model.add(tf.keras.layers.Conv2D(NUM_GEN_FILTERS*2, 3, strides=(2,2), padding='same',
                                     kernel_initializer = initializer))
    model.add(tfa.layers.InstanceNormalization(axis = 3,
                                               epsilon = 1e-5,
                                               center = True,
                                               scale = True,
                                               beta_initializer = 'random_uniform',
                                               gamma_initializer = 'random_uniform'))
    model.add(tf.keras.layers.Conv2D(NUM_GEN_FILTERS*4, 3, strides=(2,2), padding='same',
                                     kernel_initializer = initializer))
    model.add(tfa.layers.InstanceNormalization(axis = 3,
                                               epsilon = 1e-5,
                                               center = True,
                                               scale = True,
                                               beta_initializer = 'random_uniform',
                                               gamma_initializer = 'random_uniform'))
    
    return model


def generator_upsample():
    initlalizer = tf.random_normal_initializer(0., .02)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(NUM_GEN_FILTERS*2, 3, strides=(2,2), padding='same',
                                              kernel_initializer = initlalizer))
    model.add(tfa.layers.InstanceNormalization(axis = 3,
                                               epsilon = 1e-5,
                                               center = True,
                                               scale = True,
                                               beta_initializer = 'random_uniform',
                                               gamma_initializer = 'random_uniform'))
    model.add(tf.keras.layers.Conv2DTranspose(NUM_GEN_FILTERS, 3, strides=(2,2), padding='same',
                                              kernel_initializer = initlalizer))
    model.add(tfa.layers.InstanceNormalization(axis = 3,
                                               epsilon = 1e-5,
                                               center = True,
                                               scale = True,
                                               beta_initializer = 'random_uniform',
                                               gamma_initializer = 'random_uniform'))
    model.add(tf.keras.layers.Conv2D(IMG_CHANNELS, 7, strides=(1,1), padding = 'same',
                                     kernel_initializer = initlalizer))

    return model


def build_generator_resnet_9blocks(skip=False):
    inputs = tf.keras.Input(shape=[None, None, IMG_CHANNELS])

    pad_input = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    x = generator_downsample()(pad_input)

    resnet_9blocks = [resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block(),
                      resnet_block()]
    
    for block in resnet_9blocks:
        x = block(x)

    x = generator_upsample()(x)

    if skip:
        outputs = tf.nn.tanh(inputs+x)
    else:
        outputs = tf.nn.tanh(x)

    return tf.keras.Model(inputs = inputs, outputs = outputs)