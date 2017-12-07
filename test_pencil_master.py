from pencil_sketch import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color

img_name = 'pictures/ann.jpg'
img = mpimg.imread(img_name)
texture_file_name = 'textures/texture1.jpg'
texture = mpimg.imread(texture_file_name)

img_new = pencil_draw(img, texture)
#mpimg.imsave(img_name + 'skecth.png', img_new, cmap=plt.cm.gray)
plt.imshow(img_new, cmap='gray')

img_new_color = pencil_draw_color(img, texture)
mpimg.imsave(img_name + 'color_skecth.png', img_new_color, cmap=plt.cm.gray)
plt.imshow(img_new_color)
plt.show()
