from tkinter import *
from tkinter import ttk
import cv2
import os
import numpy as np
import tkinter.filedialog
from PIL import Image, ImageTk
import FaceRec
import ImageEnhance

# initial of interface
top = Tk()
top.title = 'Star similarity matching'
top.geometry('800x800')
labelContent = "please select star image and target image, then click compare"
lC = StringVar()
lC.set(labelContent)
# save selected image
pF1 = StringVar()
pF2 = StringVar()


# Standardize the size of pictures displayed on the interface
def resize(img):
    width, height = img.size
    if width == height:
        region = img
    else:
        if width > height:
            delta = (width - height) / 2
            box = (delta, 0, delta + height, height)
        else:
            delta = (height - width) / 2
            box = (0, delta, width, delta + width)
        region = img.crop(box)
    region = region.resize((250, 250), Image.ANTIALIAS)
    return region


# labels to show images and text
txt_label = Label(top, textvariable=lC, font=('Arial', 18))
txt_label.place(x=50, y=700)
load = Image.open('./test/blank.png')
load2 = resize(load)
render1 = ImageTk.PhotoImage(load2)
render2 = ImageTk.PhotoImage(load2)
img1 = Label(image=render1)
img2 = Label(image=render2)
img1.grid(row=1, column=0, pady=0, padx=8)
img2.grid(row=1, column=2, pady=0, padx=8)

# save combobox value
var = StringVar()
la = Label(top, textvariable=var)
# create combobox
enhanceType = StringVar()
chosen = ttk.Combobox(top, width=12, textvariable=enhanceType)
chosen['values'] = ('brightness', 'sharpness', 'contrast')
chosen.grid(column=2, row=3)
chosen.current(0)


# Select the picture to be compared
def choose_file():
    global img1, render1
    # select file from file folder
    selectFileName = tkinter.filedialog.askopenfilename(title='choose image')
    if selectFileName == "":
        pF1.set(os.path.abspath(selectFileName))
    else:
        pF1.set(os.path.abspath(selectFileName))
        load = Image.open(selectFileName)
        load2 = resize(load)
        render1 = ImageTk.PhotoImage(load2)
        img1.configure(image=render1)


# choose the target image function
def choose_file2():
    global img2, render2
    selectFileName = tkinter.filedialog.askopenfilename(title='choose image')
    if (selectFileName == ""):
        pF2.set(os.path.abspath(selectFileName))
    else:
        pF2.set(os.path.abspath(selectFileName))
        load = Image.open(selectFileName)
        load2 = resize(load)
        render2 = ImageTk.PhotoImage(load2)
        img2.configure(image=render2)


# create import images buttons
submit_button1 = Button(top, text="choose star image", command=choose_file, font=('Arial 16 bold'), relief=RAISED).grid(
    row=0, column=0, padx=20, pady=20)
submit_button2 = Button(top, text="choose target image", command=choose_file2, font=('Arial 16 bold'),
                        relief=RAISED).grid(row=0, column=2, padx=20, pady=20)


# make conparison of two images
def comparison():
    global txt_label
    lC.set("")
    if (pF1.get() == ""):
        lC.set('please select star image')
    elif (pF2.get() == ""):
        lC.set('please select target image')
    else:
        output, rate = FaceRec.comparison_process(pF1.get(), pF2.get())
        show_rate = np.round(rate * 100, 2)
        if show_rate > 95:
            text_label = str("They are similar enough, The similarity is: %f" % float(show_rate))
        else:
            text_label = str("They are not similar enough, The similarity is: %f" % float(show_rate))
        lC.set(text_label)
        load2 = resize(output)
        render = ImageTk.PhotoImage(load2)
        img = Label(image=render)
        img.image = render
        img.place(x=270, y=400)
    txt_label.configure(textvariable=lC)


# create comparison button
submit_button3 = Button(top, text="compare", command=comparison, font=('Arial 16 bold'), relief=RAISED).grid(row=1,
                                                                                                             column=1,
                                                                                                             padx=75,
                                                                                                             pady=20)


# image enhancement function
def enhancement():
    global img2, render2
    var.set(enhanceType.get())
    if (pF2.get() == ""):
        lC.set('please select target image')
    else:
        Img = Image.open(pF2.get())
        if enhanceType.get() == 'brightness':
            lC.set('processing brightness enhancement')
            output1 = ImageEnhance.brightness(Img)

        elif enhanceType.get() == 'sharpness':
            lC.set('processing sharpness enhancement')
            output1 = ImageEnhance.sharpness(Img)

        elif enhanceType.get() == 'contrast':
            lC.set('processing contrast enhancement')
            output1 = ImageEnhance.contrast(Img)
        Img = output1
        output = resize(output1)
        render2 = ImageTk.PhotoImage(output)
        img2.configure(image=render2)
    txt_label.configure(textvariable=lC)


enhance_button = Button(top, text="Enhance", command=enhancement, font=('Arial 16 bold'), relief=RAISED).grid(row=2,
                                                                                                              column=2,
                                                                                                              pady=10)

top.mainloop()
