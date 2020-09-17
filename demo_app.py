import os
import cv2
import uuid
import tkinter as tk 
from tkinter import messagebox as tkMessageBox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from predict import predict_image
from tkinter import W, E

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

def upload_photo():
    global panelA, filename

    filename = askopenfilename()
    opt_text = "Filename: " + filename
    lbl_upload.configure(text=opt_text)
    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    if width > 300:
        img = cv2.resize(img, (350,300))
    im = Image.fromarray(img)
    image = ImageTk.PhotoImage(image=im)

    if panelA is None:    
        panelA = tk.Label(window, image=image)
        panelA.image = image
        # panelA.grid(row=2, column=1, sticky=E)
        panelA.pack(side="top", padx=10, pady=10)
    else:
        panelA.configure(image=image)
        panelA.image = image

def open_model():
    global model
    model = askopenfilename()
    opt_text = "Classification Model: " + model
    lbl_model.configure(text=opt_text)

def open_model_pca():
    global pca
    pca = askopenfilename()
    opt_text = "PCA Model: " + pca
    lbl_pca.configure(text=opt_text)

def show_result():
    global panelB, filename, pca, model

    n_pca = int(n.get())
    label = predict_image(filename, n_pca, pca, model)

    result = "Label:" + str(label)
    lbl_predict.configure(text=result)
    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    if width > 300:
        img = cv2.resize(img, (350,300))
    im = Image.fromarray(img)
    # image = ImageTk.PhotoImage(image=im)

    # get bbox
    faces = face_cascade.detectMultiScale(img)
    print(faces)
    for x,y,w,h in faces:
        bbox = (x,y,w,h)
    print(bbox[0], bbox[1], bbox[2], bbox[3])
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)
    print("Label %d" % label)
    
    if label == 0:
        # print("%d : No Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 0", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 1:
        # print("%d : Has Angelman Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 1", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 2:
        # print("%d : Has Apert Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 2", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 3:
        # print("%d : Has Cornelia De Lange Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 3", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 4:
        # print("%d : Has Down Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 4", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 5:
        # print("%d : Has FragileX Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 5", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    elif label == 6:
        # print("%d : Has Williams Syndrome" % (label))
        cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
        cv2.putText(img, "Label 6", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    # if label==0:
    #     cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
    #     cv2.putText(img, "Label 0: No Syndrome", (x-4, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    # elif label==1:
    #     cv2.rectangle(img, (x-5,y-30), (x+w+5,y), (0,0,255), -1)
    #     cv2.putText(img, "Label 1: Syndrome", (x-4, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # im = Image.fromarray(img)
    # imgtk = ImageTk.PhotoImage(image=im)
    # if panelB is None:
    #     panelB = tk.Label(window, image=imgtk)
    #     panelB.image = imgtk
    #     panelB.pack(side="right", padx=10, pady=10)
    # else:
    #     panelB.configure(image=imgtk)
    #     panelB.image = imgtk

def change_dropdown(*args):
    print(w.get())

def on_closing():
    if tkMessageBox.askokcancel("QUIT", "Do you want to quit?"):
        window.destroy()

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Demo TA 13516142")
    window.geometry("750x750")
    window.rowconfigure(0, minsize=10, weight=1)
    window.columnconfigure(0, minsize=10, weight=1)

    panelA = None
    panelB = None
    
    lbl_upload = tk.Label(window, text="Uploaded Filename")
    # lbl_upload.grid(row=0, column=2, sticky=W)
    lbl_upload.pack(side="top")

    btn_upload = tk.Button(window, text="Upload Image", command=upload_photo)
    # btn_upload.grid(row=0, column=1, sticky=W)
    btn_upload.pack(side="top")
    
    # canvas = tk.Canvas(window, width=1000, height=500)
    # canvas.pack()

    lbl_model = tk.Label(window, text="Classification Model")
    # lbl_model.grid(row=10, column=2, sticky=W)
    lbl_model.pack(side="bottom")
    # lbl_model.config(font=('helvetica', 14))
    # canvas.create_window(500, 500, window=lbl_model)
    
    btn_model = tk.Button(window, text="Upload Model", command=open_model)
    # btn_model.grid(row=10, column=1, sticky=W)
    btn_model.pack(side="bottom")
    # btn_model.config(font=('helvetica', 10))
    # canvas.create_window(500, 550, window=btn_model)

    lbl_n = tk.Label(window, text="N value of PCA")
    # lbl_n.grid(row=12, column=1, sticky=W)
    lbl_model.pack(side="bottom")
    n = tk.Entry(window)
    n.insert(0, 'n PCA')
    # n.grid(row=12, column=2, sticky=W)
    n.pack(side="bottom", padx=3, pady=3)
    # canvas.create_window(400, 600, window=n_pca)

    lbl_pca = tk.Label(window, text="PCA Model")
    # lbl_pca.grid(row=14, column=2, sticky=W)
    lbl_pca.pack(side="bottom", padx=2, pady=2)
    # lbl_model.config(font=('helvetica', 14))
    # canvas.create_window(600, 600, window=lbl_pca)

    btn_pca = tk.Button(window, text="Upload PCA Model", command=open_model_pca)
    # btn_pca.grid(row=14, column=1, sticky=W)
    btn_pca.pack(side="bottom", padx=2, pady=2)
    # btn_pca.config(font=('helvetica', 10))
    # canvas.create_window(600, 650, window=btn_pca)

    btn_predict = tk.Button(window, text="Predict", command=show_result, bg="green", fg="white")
    # btn_predict.grid(row=16, column=1, sticky=W)
    btn_predict.pack(side="bottom", pady=5)
    # btn_predict.config(font=('helvetica', 10))
    # canvas.create_window(500, 800, window=btn_predict)

    lbl_predict = tk.Label(window, text="Prediction Result")
    # lbl_predict.grid(row=17, column=1, sticky=W)
    lbl_predict.pack(side="bottom", pady=5)
    lbl_predict.config(font=('helvetica', 16))
    # canvas.create_window(500, 850, window=lbl_predict)

    # n_pca = tk.StringVar(window)
    # CHOICES = ['1', '3', '5', '10', '20', '30', '40', '50', '100']
    # n_pca.set(CHOICES[5]) # default option
    # w = tk.OptionMenu(window, n_pca, *CHOICES)
    # w.pack(side="bottom", padx=1, pady=1)

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()