import numpy as np
from PIL import Image
from ISR.models import RRDN

img = Image.open('images_test/biosphere7_mollweide.1505.tif')
lr_img = np.array(img)

rrdn = RRDN(weights='gans')
sr_img = rrdn.predict(lr_img)
Image.fromarray(sr_img)


"""
To predict on large images and avoid memory allocation errors, use the by_patch_of_size option for the predict method, for instance


sr_img = model.predict(image, by_patch_of_size=50)

"""