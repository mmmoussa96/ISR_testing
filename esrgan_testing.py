import numpy as np
from PIL import Image
from ISR.models import RRDN

image_name = "biosphere7_mollweide.1505.tif"
img = Image.open('images_test/' + image_name)
lr_img = np.array(img)

rrdn = RRDN(weights="gans")
sr_img = rrdn.predict(lr_img)
Image.save(Image.fromarray(sr_img), "test_output/" + image_name + "_upscaled_RRDN.tif")


"""
To predict on large images and avoid memory allocation errors, use the by_patch_of_size option for the predict method, for instance


sr_img = model.predict(image, by_patch_of_size=50)

"""