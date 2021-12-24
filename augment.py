import os
import Augmentor as augment

def data_processing(root_dir: str):
  data = augment.Pipeline(root_dir)
  data.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
  data.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
  data.skew(probability=0.5, magnitude=0.5)
  data.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
  data.crop_random(probability=0.5, percentage_area=0.9, randomise_percentage_area=True)
  data.sample(1500)

data_processing('Training Data')