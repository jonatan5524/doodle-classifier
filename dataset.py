import numpy as np

files = {
  "Apple": "full_numpy_bitmap_apple.npy",
  "Banana": "full_numpy_bitmap_banana.npy",
  "Carrot": "full_numpy_bitmap_carrot.npy",
  "Pizza": "full_numpy_bitmap_pizza.npy",
}

classes = {
    "Apple": [1,0,0,0],
    "Banana": [0,1,0,0],
    "Carrot": [0,0,1,0],
    "Pizza": [0,0,0,1],
}

def load_data(data_items = 1000, training_data_split = 0.8):
    if training_data_split >= 1 and training_data_split <= 0:
        raise ValueError("The training_data_split must be between 0 and 1")

    training_data = []
    test_data = []

    for label, file in files.items():
        data = np.load(f"./dataset/{file}")
        data = data / 255.0

        if len(data) < data_items:
            raise ValueError(f"The data_items must be lower then the data items in each dataset, \
            in {label} dataset there is {len(data)} items and the data_items value given is {data_items}") 

        dataset = list(map(lambda image: (label, image), data))
        training_data += dataset[: int(data_items * training_data_split)]
        test_data += dataset[int(data_items * training_data_split) : data_items]

    return training_data, test_data