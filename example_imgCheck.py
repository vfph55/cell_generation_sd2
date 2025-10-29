
import numpy as np
from train import CellDataset
import cv2


file_path = "/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_1.csv"

dataset = CellDataset(file_path)
random_index = np.random.randint(0,len(dataset))



