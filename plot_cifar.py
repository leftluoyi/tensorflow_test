import tkinter as tk
import numpy as np
import random

pixsize = 10

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("dataset/cifar/data_batch_1")
piexls = data[b'data'][random.randint(1, 9999)]

root = tk.Tk()
canvas = tk.Canvas(root, bg="white", height=32 * pixsize, width=32 * pixsize)


inx = 0;
for i in range(0, 32 * pixsize, pixsize):
    for j in range(0, 32 * pixsize, pixsize):
        tcol = "#%02x%02x%02x" % (piexls[inx], piexls[inx + 1024], piexls[inx + 2048])
        canvas.create_rectangle(j, i, j + pixsize - 1, i + pixsize - 1, fill=tcol, outline=tcol)
        inx = inx + 1

# canvas.create_line(0,50,30,300);

canvas.pack()
root.mainloop()