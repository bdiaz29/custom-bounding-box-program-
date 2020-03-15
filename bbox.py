from tkinter import *
import PIL
from PIL import Image, ImageGrab, ImageTk
import numpy as np
from tkinter import filedialog
import os.path
from os import path
import xlwt
from tkinter import messagebox
import cv2
import os.path
import glob
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import imutils
import xlwt

vidcap = [None]
wb = xlwt.Workbook()
ws = wb.add_sheet('data')
# .grid(column=, row=)
window = Tk()
window.title("frame extracting application")
window.geometry('1300x1000')
source = ""
destination = ""
name = ""
sour = ""
state = 0
counter = 0
counter2 = 0
file_list = [None]
model = [None]
loaded_image = [None]
boxed_image = [None]

destination_set = False
source_set = False
nn_set = False
started = False
vid_sucess = True
frame_skip = 0
xmin = 0
xmax = 1
ymin = 0
ymax = 1
toggle = 0
angle = 0

x_prev = 0
y_prev = 0

# points
p1_x = 0
p2_x = 0
p1_y = 0
p2_y = 0


def pic_left_mouse(event):
    global xmin, xmax, ymin, ymax
    global toggle, p1_x, p2_x, p1_y, p2_y

    if toggle == 0:
        p1_x = event.x
        p1_y = event.y
        toggle = 1
    elif toggle == 1:
        p2_x = event.x
        p2_y = event.y
        xmin = min(p1_x, p2_x)
        ymin = min(p1_y, p2_y)
        xmax = max(p1_x, p2_x)
        ymax = max(p1_y, p2_y)

        toggle = 0
    else:
        print("something went wrong with toggle")

    update()


def calculate_iou(target_boxes, pred_boxes):
    xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
    yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
    xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
    yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
    interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
    boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def custom_loss(y_true, y_pred):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)
    return mse + (1 - iou)


def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)


def assign_source_img_folder():
    global source
    global state
    global file_list
    global counter
    global source_set
    counter = 0
    source = filedialog.askdirectory() + "/"
    file_list = [None]
    file_list = glob.glob(source + '*.jpg')
    file_list.extend(glob.glob(source + '*.jpeg'))
    file_list.extend(glob.glob(source + '*.png'))
    source_set = True
    state = 0
    counter = 0


def assign_source_video():
    global source
    global state
    global counter
    global source_set
    global vidcap, vid_sucess
    vid_sucess = True
    counter = 0
    source = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(("mp4", "*.mp4"), ("avi", "*.avi"), ("all files", "*.*")))
    vidcap = cv2.VideoCapture(source)
    source_set = True
    state = 1
    counter = 0


def assign_destination():
    global destination
    global name
    global destination_set
    name = project_name_txt.get()
    dest_String = filedialog.askdirectory(initialdir="E:/machine learning")
    destination = dest_String + "/_problem_images" + name + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)
    destination_set = True


def assign_nn():
    global nn_set
    global model
    model_source = filedialog.askopenfilename(initialdir="E:/machine learning/saved models", title="Select file",
                                              filetypes=(("h5", "*.h5"), ("all files", "*.*")))
    model = load_model(model_source, custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric})
    nn_set = True


def leftKey(event):
    global counter, state
    if state == 0:
        counter = counter - 2
    next()
    update()


def rightKey(event):
    print("right key not in use")


def register_right_click(event):
    global x_prev, y_prev
    x_prev = event.x
    y_prev = event.y
    # print("pressed right mouse", str(x_prev), str(event.x))


def change_angle(event):
    global x_prev, y_prev, angle
    x_diff = int(((x_prev - event.x) / 800) * 90)
    angle = x_diff
    update()


def start():
    global state
    global file_list
    global source
    global destination
    global source_set
    global destination_set
    global state
    global vidcap
    global loaded_image
    global nn_set
    global counter
    global started
    global frame_skip
    global name
    global ws
    # start writting to excel
    ws.write(0, 0, "ID")
    ws.write(0, 1, "xmin")
    ws.write(0, 2, "ymin")
    ws.write(0, 3, "xmax")
    ws.write(0, 4, "ymax")
    ws.write(0, 5, "P")

    picture_lbl.bind("<ButtonRelease-1>", pic_left_mouse)
    picture_lbl.bind("<B3-Motion>", change_angle)
    picture_lbl.bind("<ButtonPress-3>", register_right_click)
    picture_lbl.bind("<B1-Motion>", draw_crosshair)

    window.bind('<Left>', leftKey)
    window.bind('<Right>', rightKey)
    frame_skip_txt.configure(state="disabled")
    project_name_txt.configure(state="disabled")
    frame_skip = int(frame_skip_txt.get())
    name = project_name_txt.get()
    if not destination_set:
        print("set destination")
        return
    if not source_set:
        print("set source")
        return
    # 0 for image folder
    # 1 for video file
    if state == 0:
        image = cv2.imread(file_list[0])
        resized = imutils.resize(image, height=800)
        loaded_image = resized
        counter = counter + 1

    elif state == 1:
        success, image = vidcap.read()
        resized = imutils.resize(image, height=800)
        loaded_image = resized
        counter = counter + 1
    else:
        print("something went wrong with the state it is currently", str(state))
    started = True

    if nn_set:
        nn_boxing()
    else:
        bounds_non_nn()

    update()


def bounds_non_nn():
    global loaded_image, xmin, ymin, xmax, ymax
    shape = loaded_image.shape
    height = shape[0]
    width = shape[1]
    xmin = 0
    ymin = 0
    xmax = width
    ymax = height


def next():
    global state
    global file_list
    global source
    global destination
    global source_set
    global destination_set
    global state
    global vidcap
    global loaded_image
    global nn_set
    global counter
    global started
    global frame_skip
    global vid_sucess
    global vidcap
    global toggle, angle
    toggle = 0
    angle = 0
    frame_skip = int(frame_skip_txt.get())
    vid_counter = 0
    if not destination_set:
        print("set destination")
        return
    if not source_set:
        print("set source")
        return
    if not started:
        print("press start")
        return
    if state == 0:
        image = cv2.imread(file_list[counter])
        resized = imutils.resize(image, height=800)
        loaded_image = resized
        counter = counter + 1

    elif state == 1:
        if not vid_sucess:
            print("video not sucess")
            return
        success = True
        while (success and vid_counter <= frame_skip):
            success, image = vidcap.read()
            vid_sucess = success
            vid_counter = vid_counter + 1
        if not vid_sucess:
            print("video not sucess")
            return
        # loaded_image = image
        resized = imutils.resize(image, height=800)
        loaded_image = resized
        counter = counter + 1
    else:
        print("something went wrong with the state it is currently", str(state))
    if nn_set:
        nn_boxing()
    else:
        bounds_non_nn()
    update()


def disable_txt(event):
    frame_skip_txt.configure(state="disabled")
    project_name_txt.configure(state="disabled")


def enable_frame_skip_txt(event):
    frame_skip_txt.configure(state="normal")


def enable_project_name_txt(event):
    project_name_txt.configure(state="normal")


def resize_image_and_points(img, xmin, ymin, xmax, ymax, target_size):
    shape = img.shape
    original_height = shape[0]
    original_width = shape[1]
    target_height=target_size[0]
    target_width=target_size[1]
    y_scale=target_height/original_height
    x_scale=target_width/original_width

    x1=int(xmin*x_scale)
    y1=int(ymin*y_scale)
    x2=int(xmax*x_scale)
    y2=int(ymax*y_scale)
    resized = cv2.resize(img, (target_height,target_width), interpolation=cv2.INTER_AREA)
    return resized,x1,y1,x2,y2

def save():
    global counter, counter2
    global name
    global destination
    global loaded_image
    global xmin, ymin, xmax, ymax
    global ws, wb
    global name, angle
    if not destination_set:
        print("set destination")
        return
    if not source_set:
        print("set source")
        return
    if not started:
        print("press start")
        return
    seperated = np.array(loaded_image)
    boxdrawn = imutils.rotate(seperated, angle)
    #resize image to managable size
    save_img,x1,y1,x2,y2=resize_image_and_points(boxdrawn, xmin, ymin, xmax, ymax, [600,600])
    #save_img=cv2.circle(save_img, (x1,y1), 5,  (255, 0, 0) , 2)
    #save_img=cv2.circle(save_img, (x2, y2), 5, (0, 255, 0), 2)
    cv2.imwrite(destination + "/" + name + str(counter2) + ".jpeg", save_img)
    img_shape = loaded_image.shape
    height = img_shape[0]
    width = img_shape[1]

    # xmin_scaled = xmin / width
    # ymin_scaled = ymin / height
    # xmax_scaled = xmax / width
    # ymax_scaled = ymax / height

    ws.write(counter2 + 1, 0, str(destination + "/" + name + str(counter2) + ".jpeg"))
    ws.write(counter2 + 1, 1, x1)
    ws.write(counter2 + 1, 2, y1)
    ws.write(counter2 + 1, 3, x2)
    ws.write(counter2 + 1, 4, y2)
    ws.write(counter2 + 1, 5, 0)
    counter2 = counter2 + 1

    wb.save(destination + "/" + name + "_data.xls")


def save_next():
    save()
    next()


def save_load_next2(event):
    save_next()


def load_next2(event):
    next()


window.bind("<space>", save_load_next2)
window.bind("<p>", load_next2)


def update():
    global picture_lbl
    global loaded_image
    global frame_skip
    global xmin, ymin, xmax, ymax
    global boxed_image
    global angle
    frame_skip = int(frame_skip_txt.get())
    # keetp an image seperated from loaded image but equal
    seperated = np.array(loaded_image)
    boxdrawn = imutils.rotate(seperated, angle)
    boxdrawn = cv2.rectangle(boxdrawn, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    boxed_image = boxdrawn
    img = np.array(boxed_image)
    img = np.uint8(img)
    img_shape = np.shape(img)
    height = img_shape[0]
    width = img_shape[1]
    img2 = img[0:height, 0:width, [2, 1, 0]]
    img3 = Image.fromarray(img2)
    tk_img = ImageTk.PhotoImage(img3)
    picture_lbl.configure(image=tk_img)
    picture_lbl.image = tk_img


def draw_crosshair(event):
    global picture_lbl
    global loaded_image
    global frame_skip
    global xmin, ymin, xmax, ymax
    global boxed_image, angle
    x = event.x
    y = event.y
    seperated = np.array(loaded_image)
    img_shape = loaded_image.shape
    height = img_shape[0]
    width = img_shape[1]
    # draw bounding box
    boxdrawn = imutils.rotate(seperated, angle)
    boxdrawn = cv2.rectangle(boxdrawn, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    # draw lines
    # draw the x line
    cv2.line(boxdrawn, (0, y), (width, y), (0, 0, 0), 1)
    # draw the y line
    cv2.line(boxdrawn, (x, 0), (x, height), (0, 0, 0), 1)

    boxed_image = boxdrawn
    img = np.array(boxed_image)
    img = np.uint8(img)

    img2 = img[0:height, 0:width, [2, 1, 0]]
    img3 = Image.fromarray(img2)
    tk_img = ImageTk.PhotoImage(img3)
    picture_lbl.configure(image=tk_img)
    picture_lbl.image = tk_img


def nn_boxing():
    global model
    global loaded_image
    global boxed_image
    global xmin, ymin, xmax, ymax
    update()

    # seperated_img=Image.fromarray(seperated)
    # load and preprocess image
    img = np.array(loaded_image)
    img_shape = np.shape(img)
    height = img_shape[0]
    width = img_shape[1]
    img_slice = img[0:height, 0:width, [2, 1, 0]]
    im = Image.fromarray(img_slice)
    im2 = im.resize((224, 224))
    im3 = np.array(im2)
    img_scaled = im3 / 255
    img_dims = np.expand_dims(img_scaled, axis=0)
    # put into nerual network
    pred = model.predict(img_dims)

    x1 = (pred[0, 0])
    y1 = (pred[0, 1])
    x2 = (pred[0, 2])
    y2 = (pred[0, 3])

    # limits put in for any unusual behavior and cast as integers
    x1, y1, x2, y2 = limit(x1, y1, x2, y2, width, height)

    xmin = x1
    ymin = y1
    xmax = x2
    ymax = y2
    p = 0


def limit(x1, y1, x2, y2, X, Y):
    x1 = x1 * X
    y1 = y1 * Y
    x2 = x2 * X
    y2 = y2 * Y
    if x1 >= X:
        x1 = X - 1

    if x2 >= X:
        x2 = X - 1

    if y1 >= Y:
        y1 = Y - 1

    if y2 >= Y:
        y2 = Y - 1

    ############
    if x1 < 0:
        x1 = 0

    if x2 < 0:
        x2 = 0

    if y1 < 0:
        y1 = 0

    if y2 < 0:
        y2 = 0

    x1 = int(min(x1, x2))
    x2 = int(max(x1, x2))
    y1 = int(min(y1, y2))
    y2 = int(max(y1, y2))

    if x1 == x2:
        x1 = 0
        x2 = X - 1

    if y1 == y2:
        y1 = 0
        y2 = Y - 1

    return x1, y1, x2, y2


file_frame = Frame(master=window)
file_frame.grid(column=0, row=0)

load_img_folder_btn = Button(file_frame, text="load image folder", command=assign_source_img_folder)
load_img_folder_btn.grid(column=0, row=0)

load_video_btn = Button(file_frame, text="load video", command=assign_source_video)
load_video_btn.grid(column=0, row=1)

# load_video_folder_btn = Button(file_frame, text="load video folder")
# load_video_folder_btn.grid(column=0, row=2)

save_dir_btn = Button(file_frame, text="save directory", command=assign_destination)
save_dir_btn.grid(column=0, row=3)

blank = np.zeros((800, 800, 3))
blank = np.uint8(blank)
blank_img = Image.fromarray(blank)
blank2 = ImageTk.PhotoImage(blank_img)
picture_lbl = Label(file_frame, image=blank2, cursor='crosshair')
picture_lbl.grid(column=0, row=5)

frame_skip_lbl = Label(file_frame, text="frame skip")
frame_skip_lbl.grid(column=1, row=0)

frame_skip_txt = Entry(file_frame)
frame_skip_txt.grid(column=2, row=0)
frame_skip_txt.insert(END, "0")
frame_skip_txt.bind("<Return>", disable_txt)
frame_skip_txt.bind("<Button 1>", enable_frame_skip_txt)

project_name_lbl = Label(file_frame, text="project name")
project_name_lbl.grid(column=1, row=1)

load_nn_btn = Button(file_frame, text="load .h5 file", command=assign_nn)
load_nn_btn.grid(column=3, row=1)

project_name_txt = Entry(file_frame)
project_name_txt.grid(column=2, row=1)
project_name_txt.bind("<Return>", disable_txt)
project_name_txt.bind("<Button 1>", enable_project_name_txt)

next_btn = Button(file_frame, text="next", command=next)
next_btn.grid(column=0, row=4)

save_btn = Button(file_frame, text="save", command=save)
save_btn.grid(column=1, row=4)

save_next_btn = Button(file_frame, text="save and next", command=save_next)
save_next_btn.grid(column=2, row=4)

start_btn = Button(file_frame, text="start", command=start)
start_btn.grid(column=2, row=5)

window.bind("<space>", save_load_next2)
window.bind("<m>", load_next2)

window.mainloop()
