from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import glob

path = glob.glob('./mamografias/**/*.png')

data_index = 0
data_len = len(path)
img = Image.open(path[data_index])
img = img.resize((256, 256))
curr_img = plt.imshow(np.asarray(img), cmap='Greys_r')
c0 = 3
axcolor = 'lightgoldenrodyellow'

# create Button
nextax = plt.axes([0.7, 0.020, 0.1, 0.04])  # position of button
button_next = Button(nextax, 'next', hovercolor='0.975')
prevax = plt.axes([0.2, 0.020, 0.1, 0.04])  # position of button
button_prev = Button(prevax, 'prev', hovercolor='0.975')
resetax = plt.axes([0.8, 0.020, 0.1, 0.04])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# create slider
sliderax = plt.axes([0.15, -0.018, 0.65, 0.03], facecolor=axcolor) #position of slider
scontrast = Slider(sliderax, 'Contrast', 2.1, 30.0, valinit=c0)


# # adjusting contrast
# def updateContrast(image):
#     minColor = np.amin(image)
#     maxColor = np.amax(image)
#     https://stackoverflow.com/questions/48406578/adjusting-contrast-of-image-purely-with-numpy/50602577#50602577

def update(val):
    global curr_img
    contrast = scontrast.val
    print(path[val])
    img = Image.open(path[val])
    img = img.resize((256, 256))
    curr_img.set(data=img)
    plt.draw()
    scontrast.on_changed(update)

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

def reset(event):
    scontrast.reset()
button_reset.on_clicked(reset)



button_prev.on_clicked(prev)
button_next.on_clicked(next)
plt.show()
