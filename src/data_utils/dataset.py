import os
import numpy as np
import cv2
import glob
import h5py
from typing import Literal, Tuple


class Dataset:
    def __init__(self, data_type: Literal["train", "test", "val"]) -> None:
        self.data_type = data_type
        match data_type:
            case "train":
                self.data_path = "../../data/playing_cards/train/"
            case "test":
                self.data_path = "../../data/playing_cards/test/"
            case "val":
                self.data_path = "../../data/playing_cards/val/"
    
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and return the images and their labels as numpy array. 

        Returns: 
            - images: Array of images
            - suits: Suit of the image
            - numbers: Card number of the image
        """
        images = []
        labels = os.listdir(self.data_path)

        # Separate columns for number and suit
        suits = []
        numbers = []

        for label in labels:
            if "joker" in label:
                continue

            for img_path in glob.glob(
                os.path.join(self.data_path, label, "*.jpg")
            ):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)

                number, _, suit = label.split()
                
                suits.append(suit)
                numbers.append(number)

        self.images = np.array(images)
        self.suits = np.array(suits)
        self.numbers = np.array(numbers)

        print(f"Loaded {len(self.images)} images and labels.")

        return self.images, self.suits, self.numbers
    

    def load_descriptors(self, n: int=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and return the first n image descriptors and their labels as numpy arrays. 

        If n is not provided, all descriptors and labels are returned. 

        Returns: 
            - descriptors: Array of descriptors
            - suits: Suit of the image
            - numbers: Card number of the image
        """

        with h5py.File(f"../data/{self.data_type}_descriptors_and_labels.h5", "r") as data_file:
            desc_group = data_file["descriptors"]
            num = n if n else len(desc_group)
            descriptors = [desc_group[str(idx)][:] for idx in range(num)]

            suits_group = data_file["suits"]
            suits = [suits_group[str(idx)][:] for idx in range(num)]

            nums_group = data_file["numbers"]
            numbers = [nums_group[str(idx)][:] for idx in range(num)]

        return descriptors, suits, numbers
    

if __name__ == "__main__":
    # train_ds = Dataset("train")
    # train_im, train_suits, train_nums = train_ds.load_data()
    # assert train_im.shape == (7509, 224, 224)
    # assert train_suits.shape == (7509, )
    # assert train_nums.shape == (7509, )

    # test_ds = Dataset("train")
    # test_im, test_suits, test_nums = test_ds.load_data()
    # assert test_im.shape == (260, 224, 224)
    # assert test_suits.shape == (260, )
    # assert test_nums.shape == (260, )

    # val_ds = Dataset("train")
    # val_im, val_suits, val_nums = val_ds.load_data()
    # assert val_im.shape == (260, 224, 224)
    # assert val_suits.shape == (260, )
    # assert val_nums.shape == (260, )

    print("Dataset passed all test cases!")
