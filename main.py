from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import glob

path = glob.glob('./mamografias/DleftCC/*.png')
# path = glob.glob('./mamografias/**/*.png')

data_index = 0
data_len = len(path)
img = Image.open(path[data_index])
img = img.resize((256, 256))
curr_img = plt.imshow(np.asarray(img), cmap='Greys_r')

# create Button
nextax = plt.axes([0.7, 0.025, 0.1, 0.04])  # position of button
button_next = Button(nextax, 'next', hovercolor='0.975')
prevax = plt.axes([0.2, 0.025, 0.1, 0.04])  # position of button
button_prev = Button(prevax, 'prev', hovercolor='0.975')


def update(val):
    global curr_img
    img = Image.open(path[val])
    img = img.resize((256, 256))
    curr_img.set(data=img)
    plt.draw()


def next(event):
    global data_index
    global data_len
    data_index = (data_index + 1) % data_len
    update(data_index)


def prev(event):
    global data_index
    global data_len
    data_index = (data_index - 1) % data_len
    update(data_index)


button_prev.on_clicked(prev)
button_next.on_clicked(next)
plt.show()
