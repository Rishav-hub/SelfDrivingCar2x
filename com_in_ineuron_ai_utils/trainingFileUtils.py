import scipy.misc
import random
import matplotlib.image as mpimg
import numpy as np
import cv2


def image_generator():
    xs = []
    ys = []

    # points to the end of the last batch
    training_len_counter = 0
    validation_len_counter = 0

    # read ORIGINALdata.txt
    with open("./dataSets/TrainTestDataSmall2_2/data.txt") as f:
        for line in f:
            xs.append("./dataSets/TrainTestDataSmall2_2/" + line.split()[0])
            # the paper by Nvidia uses the inverse of the turning radius,
            # but steering wheel angle is proportional to the inverse of turning radius
            # so the steering wheel angle in radians is used as the output
            ys.append(float(line.split()[1]) * scipy.pi / 180)

    # get number of images
    num_images = len(xs)

    train_xs = xs[:int(len(xs) * 0.8)]
    train_ys = ys[:int(len(xs) * 0.8)]

    val_xs = xs[-int(len(xs) * 0.2):]
    val_ys = ys[-int(len(xs) * 0.2):]

    num_train_images = len(train_xs)
    num_val_images = len(val_xs)

    print(num_train_images)
    print(num_val_images)
    return train_xs, train_ys, val_xs, val_ys, num_train_images, num_val_images
def img_preprocess(img):

    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def batch_generator(image_paths, steering_ang, batch_size, istraining):

  while True:
    batch_img = []
    batch_steering = []

    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)

      if istraining:
        steering = steering_ang[random_index]
        im = mpimg.imread(image_paths[random_index])

      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      im = img_preprocess(im)
      
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))
