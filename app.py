from tkinter import filedialog,Tk,Button,Canvas,LEFT,colorchooser,Entry,Label
from PIL import ImageTk,Image  
from main import *
import cv2


window=Tk()
window.geometry("500x500")

# global params
filename = ""
img = None
image = None
rgb = None
rgb_image = None
rgb_img = None

canvas = Canvas(window, width = 300, height = 300)  
canvas.pack(expand='yes')  

def openImage():
    global filename, img, image,canvas
    filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Image Files",["*.jpg","*.png","*.jfif"]),("all files","*.*")))  
    img = Image.open(filename)
    w,h = img.width,img.height
    w = int(300 * (w/h))
    img = img.resize((w, 300), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)   
    image = canvas.create_image(0,0, anchor = 'nw', image = img)

def save():
    global rgb
    file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    file = file.name
    if file:
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(file,rgb)

def get_rgb_tuple(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def colorize():
    global filename,image,canvas,rgb,rgb_image,rgb_img,e1
    _,color = colorchooser.askcolor(title='Select new hair color')
    res = get_prediction(filename,load_image(filename,True).shape)

    image_original = load_image(filename)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 1, 1))
    erosion_dst = cv2.erode(res, element)
    
    orig_shape = erosion_dst.shape
    shape = (orig_shape[0],orig_shape[1],3)
    r,g,b = get_rgb_tuple(color)
   
    mask = np.zeros(shape)

    mask[:,:,0] = np.full(orig_shape,r,np.uint8)
    mask[:,:,1] = np.full(orig_shape,g,np.uint8)
    mask[:,:,2] = np.full(orig_shape,b,np.uint8)


    thresh = np.average(erosion_dst) + 0.1
    if e1.get() != '':
        thresh = float(e1.get())
    else:
        e1.insert(0,thresh)
    
    real_img = image_original.copy()
    blend_factor = 0.35
    avg_val = np.average(erosion_dst)
    for i in range(orig_shape[0]):
        for j in range(orig_shape[1]):
            if erosion_dst[i,j] > thresh:
                #retain
                real_img[i,j,0] = r *blend_factor  + real_img[i,j,0]*(1-blend_factor)
                real_img[i,j,1] = g * blend_factor + real_img[i,j,1]*(1-blend_factor)
                real_img[i,j,2] = b* blend_factor + real_img[i,j,2]*(1-blend_factor)
            else:
                mask[i,j,0] = 0
                mask[i,j,1] = 0
                mask[i,j,2] = 0

    real_img = cv2.medianBlur(real_img,3)

    display(real_img)


openBtn = Button(window, text ="Open", command = openImage)
openBtn.pack(side=LEFT,padx=10,pady=10)

# Convert Button
colorizeBtn = Button(window, text ="Colorize", command = colorize)
colorizeBtn.pack(side=LEFT,padx=10,pady=10)

# Save Button
saveBtn = Button(window, text ="Save", command = save)
saveBtn.pack(side=LEFT,padx=10,pady=10)

# text box for thresholding
l = Label(window,text="Threshold")   
l.pack(side=LEFT,padx=10,pady=10)

e1 = Entry(window)
e1.pack(side=LEFT,padx=10,pady=10)

if __name__ == "__main__":
    window.mainloop()