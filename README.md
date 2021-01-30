# doodle-classifier

doodle classifier using the Quick, Draw! dataset.

model trained to classify 4 classes:

- Apple
- Pizza
- Carrot
- Banana

# The model:

CNN model using tensorflow:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 28)        280
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 28)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 56)        14168
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 56)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 56)          28280
_________________________________________________________________
flatten (Flatten)            (None, 504)               0
_________________________________________________________________
dense (Dense)                (None, 4)                 2020
=================================================================
Total params: 44,748
Trainable params: 44,748
Non-trainable params: 0
_________________________________________________________________
```

# Training:

The model trained using each of the classes dataset npy file from the dataset.
While trained the npy files located in the dataset folder in the root directory.

# Resources:

[Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

[tensorflow docs](https://www.tensorflow.org/)
