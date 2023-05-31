# import numpy as np
# import matplotlib.pyplot as plt
# labels = np.load('labels.npy')
# bin_edges = [0,20,40,60,80,100,120]
# plt.hist(labels, bins= bin_edges, edgecolor= 'black')
# plt.xlabel('Intervals')
# plt.ylabel('Frequency')
# plt.title('Frequency of Intervals')
# plt.show()


# import netCDF4,os
# files = os.listdir('Satellite Imagery')
# for i in files:
#     raw_data = netCDF4.Dataset('Satellite Imagery/' + i)
#     ir_data = raw_data.variables['IRWIN'][0]
#     print(ir_data.shape)
#     break

import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import metrics
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import load_model

def augment_images(images, labels):

    # Create generators to augment images
    from tensorflow.keras.preprocessing import image
    flip_generator = image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )
    rotate_generator = image.ImageDataGenerator(
        rotation_range=360,
        fill_mode='nearest'
    )

    # Accumulate augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Loop each images in the set to augment
    for i in range(len(images)):

        # Reshape image for generator
        image = np.reshape(images[i], (1, images[i].shape[0], images[i].shape[1], 1))
        label = labels[i]

        # Reset the number of augmented images have been created to zero
        num_new_images = 0

        # Generate 2 new images if the image is of a tropical cyclone between 50 and 75 knots
        if 50 < label < 75:
            for batch in flip_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 2:
                    break

        # Generate 6 new images if the image is of a tropical cyclone between 75 and 100 knots
        elif 75 < label < 100:
            for batch in rotate_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 6:
                    break

        # Generate 12 new images if the image is of a tropical cyclone greater than or equal to 100 knots
        elif 100 <= label:
            for batch in rotate_generator.flow(image, batch_size=1):
                gc.collect()
                new_image = np.reshape(batch[0], (batch[0].shape[0], batch[0].shape[1], 1))
                augmented_images.append(new_image)
                augmented_labels.append(label)
                num_new_images += 1
                if num_new_images == 12:
                    break


    # Convert lists of images/labels into numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    augmented_labels = np.append(augmented_labels,labels,0)
    bin_edges = [0,20,40,60,80,100,120]
    plt.hist(augmented_labels, bins= bin_edges, edgecolor= 'black')
    plt.xlabel('Intervals')
    plt.ylabel('Frequency')
    plt.title('Frequency of Intervals')
    plt.show()

images = np.load('images.npy')
labels = np.load('labels.npy')
print(len(images))
augment_images(images, labels)

