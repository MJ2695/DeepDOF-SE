import tensorflow as tf
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    return cropped_image

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = image/127.5 - 1
    
    return image

def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)

    return image

def load_image(file):
    image = tf.io.read_file(file)
    image = tf.io.decode_png(image)
    image = normalize(image)

    return image

def generate_images(model, test_input):
    prediction = model(test_input, training=False)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()