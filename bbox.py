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
import pandas as pd
import xlsxwriter

####work on crosshair
vidcap = [None]
wb = xlwt.Workbook()
ws = wb.add_sheet('data', cell_overwrite_ok=True)
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

x_prev = 0
y_prev = 0

# points
p1_x = 0
p2_x = 0
p1_y = 0
p2_y = 0

# array that holds all the images
img_arr = []
# array that holds all the image points
img_arr_points = []
img_vid = True
# what image is being outputted
output_img_index = 0
# the image that is being ouputed
output_img = np.zeros((800, 800, 3))
# whether the selected data is to be saved
save = []
# frame_skip
angle = []
# the set height and width of image
set_height = 600
set_width = 600

# what state the output image is in
output_img_state = 0
proximity_modified = False
click_state = 3


def update_img():
    global output_img_index, angle
    global img_arr, img_arr_points, save
    # if nothing loaded into image
    if img_arr == []:
        print("no images loaded yet")
        return

    length = len(img_arr)
    # print(str(length))
    if output_img_index >= length:
        output_img_index = length - 1

    txt = str(output_img_index + 1) + "/" + str(length)
    num_lbl.configure(text=txt)

    # adjust to index
    point_array = img_arr_points[output_img_index]
    # save state
    saved = save[output_img_index]
    x1 = point_array[0]
    y1 = point_array[1]
    x2 = point_array[2]
    y2 = point_array[3]
    img = img_arr[output_img_index]
    # adjust image by angle
    img2 = imutils.rotate(img, angle[output_img_index])
    # draw bounds
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (x1, y1)

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (x2, y2)

    # calor dependent on save state
    if saved:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    # Using cv2.rectangle() method
    # Draw a rectangle the borders
    img3 = cv2.rectangle(img2, start_point, end_point, color, thickness)
    # convert to rgb
    # im_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img4 = Image.fromarray(np.uint8(np.array(img3)))
    tk_img = ImageTk.PhotoImage(img4)
    picture_lbl.configure(image=tk_img)
    picture_lbl.image = tk_img


def toggle_save():
    global save, output_img_index
    if not save[output_img_index]:
        save[output_img_index] = True
    else:
        save[output_img_index] = False
    update_img()

def change_angle(event):
    global x_prev, y_prev, angle, output_img_index
    x_diff = int(((x_prev - event.x) / 800) * 90)
    angle[output_img_index] = x_diff
    draw_crosshair(event)

def proximity_change(x, y, x1, y1, x2, y2):
    x1_diff = abs(x1 - x)
    x2_diff = abs(x2 - x)
    y1_diff = abs(y - y1)
    y2_diff = abs(y - y2)

    if x1_diff <= x2_diff:
        x1 = x
    else:
        x2 = x

    if y1_diff <= y2_diff:
        y1 = y
    else:
        y2 = y

    return [x1, y1, x2, y2]


def horizontal_change(x, y, x1, y1, x2, y2):
    global click_state
    click_state = 1
    x1_diff = abs(x1 - x)
    x2_diff = abs(x2 - x)
    if x1_diff <= x2_diff:
        x1 = x
    else:
        x2 = x
    return [x1, y1, x2, y2]


def vertical_change(x, y, x1, y1, x2, y2):
    global click_state
    click_state = 2
    y1_diff = abs(y - y1)
    y2_diff = abs(y - y2)
    if y1_diff <= y2_diff:
        y1 = y
    else:
        y2 = y
    return [x1, y1, x2, y2]


def pic_left_mouse(event):
    global xmin, xmax, ymin, ymax, save, output_img_index
    global toggle, p1_x, p2_x, p1_y, p2_y, img_arr_points, img_arr
    global proximity_modified
    s = np.shape(img_arr[output_img_index])
    height = s[0]
    width = s[1]

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

        if xmin <= 0:
            xmin = 0
        if ymin <= 0:
            ymin = 0
        if xmax >= width:
            xmax = width - 1
        if ymax >= height:
            ymax = height - 1

        save[output_img_index] = True
        img_arr_points[output_img_index] = [xmin, ymin, xmax, ymax]
        toggle = 0
        save[output_img_index]=True
    else:
        print("something went wrong with toggle")

    update_img()


def pic_right_mouse(event):
    # draw_crosshair(event)
    update_img()


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
    global file_list, angle
    global source_set, save, img_arr_points
    global img_arr, path, frame_skip, file_list
    # grey out project name and frame skip
    project_name_txt.configure(state="disabled")
    frame_skip_txt.configure(state="disabled")
    counter = 0
    source = filedialog.askdirectory() + "/"
    file_list = [None]
    file_list = glob.glob(source + '*.jpg')
    file_list.extend(glob.glob(source + '*.jpeg'))
    file_list.extend(glob.glob(source + '*.png'))
    source_set = True
    state = 0
    count = 0
    length = len(file_list)
    # populate the array with new images
    print("reading image files")
    p = 1
    pp = int(length / 10)
    for j in range(length):
        if j >= p * pp:
            print(str(p) + "0% finished")
            p = p + 1
        if count == frame_skip:
            img = cv2.imread(file_list[j])
            # if the read image does not have as shape attribute
            if not hasattr(img, 'shape'):
                continue
            resized_img = imutils.resize(img, height=800)
            # convert to rgb
            im_rgb = resized_img[:, :, [2, 1, 0]]
            img_arr += [im_rgb]
            save += [False]
            s = np.shape(im_rgb)
            height = s[0]
            width = s[1]
            img_arr_points += [[0, 0, width - 1, height - 1]]
            angle += [0]
            count = 0
        else:
            count = count + 1
    print("finished reading images")
    if img_arr != []:
        update_img()


def assign_source_video():
    global source
    global state, save
    global counter, img_arr, img_arr_points
    global source_set, frame_skip
    global vidcap, vid_sucess, angle
    frame_skip = int(frame_skip_txt.get())
    vid_sucess = True
    counter = 0
    source0 = filedialog.askopenfilename(initialdir="/", title="Select file",
                                         filetypes=(
                                         ("mp4", "*.mp4"), ("avi", "*.avi"), ("webm", "*.webm"), ("all files", "*.*")))
    cap = cv2.VideoCapture(source0)
    # if not under_four_gb(cap, frame_skip):
    #   temp = "video size is " + str(vid_array_size(source, frame_skip)) + " over 4gb,video not loaded"
    #  messagebox.showwarning("Warning", "current video at frame skip is over 4gb , video not loaded, vi")
    # return
    # get the meta data
    h, w, number_of_frames = vid_meta(cap)
    source = source0
    source_set = True
    state = 1
    count = 0
    count2 = 0
    # cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    resized_frame = imutils.resize(frame, height=800)
    # convert to rgb
    im_rgb = resized_frame[:, :, [2, 1, 0]]
    img_arr += [im_rgb]
    print("turning video into frames")
    p = 1
    pp = int(number_of_frames / 10)
    while (ret):
        # tracking
        if count2 >= p * pp:
            print(str(p) + "0% finished")
            p = p + 1
        count2 = count2 + 1
        # Capture frame-by-frame
        # skip is frame skip
        ret, frame = cap.read()
        if count == frame_skip:
            # if the read image does not have as shape attribute
            if not hasattr(frame, 'shape'):
                continue
            resized_frame = imutils.resize(frame, height=800)
            # convert to rgb
            im_rgb = resized_frame[:, :, [2, 1, 0]]
            img_arr += [im_rgb]
            save += [False]
            count = 0
            s = np.shape(im_rgb)
            height = s[0]
            width = s[1]
            img_arr_points += [[0, 0, width - 1, height - 1]]
            angle += [0]
        else:
            count = count + 1
    print("finished")
    update_img()


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


# purge unsaved images
def purge_unsaved():
    global img_arr, img_arr_points, save, angle
    length = len(save)
    # deleteparts that are not saved
    count = 0
    new_img_arr = []
    new_img_arr_points = []
    new_angle = []
    new_save = []
    print("purging unsaved images")
    for i in range(length):
        if save[i]:
            new_img_arr += [img_arr[i]]
            new_img_arr_points += [img_arr_points[i]]
            new_angle += [angle[i]]
            new_save += [True]
    print("finished")
    # empty lists
    img_arr = []
    img_arr_points = []
    save = []
    angle = []

    # copy new lists
    img_arr = new_img_arr
    img_arr_points = new_img_arr_points
    angle = new_angle
    save = new_save

    update_img()


# if a nueral network is loaded
# need to scale out the numbers
def assign_nn():
    global nn_set, img_arr
    global model, img_arr_points
    model_source = filedialog.askopenfilename(initialdir="E:/machine learning/saved models", title="Select file",
                                              filetypes=(("h5", "*.h5"), ("all files", "*.*")))
    # model = load_model(model_source, custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric})
    model = tf.keras.models.load_model(model_source,
                                       custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric,
                                                       'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.3)},compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss='mse',
        metrics=['mse'])
    # make temp array to hold 224 224 images.
    length = len(img_arr)
    temp_arr = []
    print("preparing images for neural network")
    p = 1
    pp = int(length / 10)
    for i in range(length):
        if i >= p * pp:
            print(str(p) + "0% finished")
            p = p + 1
        target_size = (224, 224)
        resized_image = cv2.resize(img_arr[i], target_size)
        temp_img = np.array(resized_image) / 255
        temp_arr += [temp_img]
    temp_arr = np.array(temp_arr)
    # temp_arr=np.expand_dims(temp_arr,axis=0)
    print("feeding images into neural network")
    arr_points = model.predict(temp_arr)
    print("finished")
    del temp_arr

    for j in range(length):
        s = np.shape(img_arr[j])
        points = arr_points[j]
        height = s[0]
        width = s[1]
        x1 = int(points[0] * width)
        y1 = int(points[1] * height)
        x2 = int(points[2] * width)
        y2 = int(points[3] * height)
        img_arr_points[j] = [x1, y1, x2, y2]

    nn_set = True
    update_img()


def leftKey(event):
    global output_img_index, toggle
    toggle = 0
    if output_img_index - 1 < 0:
        print("cant go lower")
        return
    output_img_index = output_img_index - 1
    update_img()


def rightKey(event):
    global output_img_index, img_arr, toggle
    toggle = 0
    maximum = len(img_arr)
    if output_img_index + 1 >= maximum:
        print("cant go higher")
        return
    output_img_index = output_img_index + 1
    update_img()


def register_right_click(event):
    global x_prev, y_prev
    x_prev = event.x
    y_prev = event.y
    # print("pressed right mouse", str(x_prev), str(event.x))


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
    target_height = target_size[0]
    target_width = target_size[1]
    y_scale = target_height / original_height
    x_scale = target_width / original_width

    x1 = int(xmin * x_scale)
    y1 = int(ymin * y_scale)
    x2 = int(xmax * x_scale)
    y2 = int(ymax * y_scale)
    resized = cv2.resize(img, (target_height, target_width), interpolation=cv2.INTER_AREA)
    return resized, x1, y1, x2, y2


def right_click(event):
    draw_crosshair(event)


def left_click(event):
    draw_crosshair(event)


def draw_crosshair(event):
    global picture_lbl, img_arr
    global loaded_image, output_img_index, img_arr_points
    global frame_skip, click_state
    global boxed_image, angle,toggle
    x = event.x
    y = event.y

    point_array = img_arr_points[output_img_index]
    # save state
    saved = save[output_img_index]

    seperated = np.array(img_arr[output_img_index])

    point_array = img_arr_points[output_img_index]
    # save state
    saved = save[output_img_index]
    x1 = point_array[0]
    y1 = point_array[1]
    x2 = point_array[2]
    y2 = point_array[3]
    img = img_arr[output_img_index]

    # adjust image by angle
    img2 = imutils.rotate(seperated, angle[output_img_index])
    # draw bounds
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (x1, y1)

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (x2, y2)

    # calor dependent on save state
    if saved:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.rectangle() method
    # Draw a rectangle the borders
    img3 = cv2.rectangle(img2, start_point, end_point, color, thickness)
    # convert to rgb

    # tk_img = ImageTk.PhotoImage(img4)
    # picture_lbl.configure(image=tk_img)
    # picture_lbl.image = tk_img

    img_shape = np.shape(img3)
    height = img_shape[0]
    width = img_shape[1]

    cv2.line(img3, (0, y), (width, y), (0, 0, 0), 1)
    cv2.line(img3, (x, 0), (x, height), (0, 0, 0), 1)
    # im_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img4 = Image.fromarray(np.uint8(np.array(img3)))
    tk_img = ImageTk.PhotoImage(img4)
    picture_lbl.configure(image=tk_img)
    picture_lbl.image = tk_img


# returns he meta data of video
def vid_meta(vcap):
    # vcap = cv2.VideoCapture(vid)  # 0=camera
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        # print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4

        # or
        width = vcap.get(3)  # float
        height = vcap.get(4)  # float

        # print('width, height:', width, height)

        fps = vcap.get(cv2.CAP_PROP_FPS)
        # print('fps:', fps)  # float
        # print(cv2.CAP_PROP_FPS) # 5

        frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print('frames count:', frame_count)  # float
        # print(cv2.CAP_PROP_FRAME_COUNT) # 7
        return height, width, frame_count


# returns if it would be over 4 gigabytes
def under_four_gb(cap, frame_skip):
    # 4 gigabytes in bits
    limit = 32000000000
    height, width, frame_count = vid_meta(cap)
    projected_size = frame_count * height * width * 255 / (frame_skip + 1)
    if projected_size <= limit:
        return True
    else:
        return False


# returns the amount of GB video at frame skip would be
def vid_array_size(vid, frame_skip):
    height, width, frame_count = vid_meta(vid)
    global set_height, set_width
    height = set_height
    width = set_width
    projected_size = frame_count * height * width * 255 / (frame_skip * 8000000000)
    return projected_size


# done after image array populated
# populate the points
def populate_points():
    global img_arr, img_arr_points, model, nn_set
    arr_length = len(img_arr)
    if nn_set:
        # batch predicts all the bounds
        pred = model.predict(img_arr)
        for i in range(arr_length):
            img_arr_points += [pred[i]]

    else:
        for j in range(arr_length):
            img_arr_points += [[0, 0, set_height, set_width]]


def save_load_next(event):
    global save
    save[output_img_index] = True
    frame_skip_txt.configure(state='disabled')
    project_name_txt.configure(state='disabled')
    rightKey(event)


# save current iteration
# note to organize save for origin, and for destination
def save_to_file():
    global destination, save, img_arr_points, img_arr
    global name, angle
    name = project_name_txt.get()
    length = len(save)
    ix = 0
    ws.write(ix, 0, "ID")
    ws.write(ix, 1, "xmin")
    ws.write(ix, 2, "ymin")
    ws.write(ix, 3, "xmax")
    ws.write(ix, 4, "ymax")
    ws.write(ix, 5, "P")
    if destination == "":
        print("set destination")
        return
    print("saving pictures and excel file")
    p = 1
    pp = int(length / 10)
    for i in range(length):

        if i >= p * pp:
            print(str(p) + "0% finished")
            p = p + 1
        # if marked to be saved
        if save[i]:
            ix = ix + 1
            temp = destination + str(ix) + ".jpg"
            img = img_arr[i]
            img2 = imutils.rotate(img, angle[i])
            img_points = img_arr_points[i]
            shape = np.shape(img)
            height = shape[0]
            width = shape[1]
            # x_scale = 224 / width
            # y_scale = 224 / height
            # save to excel
            x1 = img_points[0] / width
            y1 = img_points[1] / height
            x2 = img_points[2] / width
            y2 = img_points[3] / height
            ws.write(ix, 0, temp)
            ws.write(ix, 1, x1)
            ws.write(ix, 2, y1)
            ws.write(ix, 3, x2)
            ws.write(ix, 4, y2)
            ws.write(ix, 5, 0)
            # convert back to bgr
            img3 = img2[:, :, [2, 1, 0]]
            #img4=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(temp, img3)
    wb.save(destination + "/" + name + "_data.xls")
    print("finished")

def save_just_xl():
    global destination, save, img_arr_points, img_arr
    global name, angle
    name = project_name_txt.get()
    length = len(save)
    ix = 0
    save_name=filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
    workbook = xlsxwriter.Workbook(destination + name + ".xlsx")
    worksheet = workbook.add_worksheet()

    worksheet.write(ix, 0, "ID")
    worksheet.write(ix, 1, "xmin")
    worksheet.write(ix, 2, "ymin")
    worksheet.write(ix, 3, "xmax")
    worksheet.write(ix, 4, "ymax")
    worksheet.write(ix, 5, "P")
    if destination == "":
        print("set destination")
        return
    print("saving pictures and excel file")
    p = 1
    pp = int(length / 10)
    for i in range(length):

        if i >= p * pp:
            print(str(p) + "0% finished")
            p = p + 1
        # if marked to be saved
        if save[i]:
            ix = ix + 1
            temp = destination + str(ix) + ".jpg"
            img = img_arr[i]
            img2 = imutils.rotate(img, angle[i])
            img_points = img_arr_points[i]
            shape = np.shape(img)
            height = shape[0]
            width = shape[1]
            # x_scale = 224 / width
            # y_scale = 224 / height
            # save to excel
            x1 = img_points[0] / width
            y1 = img_points[1] / height
            x2 = img_points[2] / width
            y2 = img_points[3] / height
            worksheet.write(ix, 0, temp)
            worksheet.write(ix, 1, x1)
            worksheet.write(ix, 2, y1)
            worksheet.write(ix, 3, x2)
            worksheet.write(ix, 4, y2)
            worksheet.write(ix, 5, 0)
            # convert back to bgr
    workbook.close()
    print("finished")

def load_xl():
    global source
    global img_arr
    global img_arr_points
    global angle,save
    source = filedialog.askopenfilename(initialdir="/", title="Select file",
                                        filetypes=(
                                            ("xls", "*.xls"), ("xlsx", "*.xlsx"), ("all files", "*.*")))
    df = pd.read_excel(source)
    df_list = df.to_numpy()
    file_list = np.delete(df_list, 0, 0)
    shape = np.shape(file_list)
    rows = shape[0]
    columns=shape[1]
    xmin_index=1
    ymin_index=2
    xmax_index=3
    ymax_index=4

    for j in range(1,columns):
        val=filename = file_list[0, j]
        if val=="xmin":
            xmin_index=j
        if val=="ymin":
            ymin_index=j
        if val=='xmax':
            xmax_index=j
        if val=='ymax':
            ymax_index=j

    p = 1
    pp = int(rows / 100)
    print("loading files from excel file")
    for i in range(rows):
        if i >= p * pp:
            print(str(p) + "% finished")
            p = p + 1

        filename = file_list[i, 0]
        im = PIL.Image.open(filename)
        img_array = np.array(im)
        # truncate the 4th dimension if png
        image_shape = np.shape(img_array)
        img_height = image_shape[0]
        img_width = image_shape[1]
        xmin = int(float(file_list[i, xmin_index]) * img_width)
        ymin = int(float(file_list[i, ymin_index]) * img_height)
        xmax = int(float(file_list[i, xmax_index]) * img_width)
        ymax = int(float(file_list[i, ymax_index]) * img_height)

        img_arr += [img_array]
        img_arr_points += [[xmin, ymin, xmax, ymax]]
        angle+=[0]
        save+=[True]
    print("finished loading files")
    update_img()

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

num_lbl = Label(file_frame, text="blank")
num_lbl.grid(column=0, row=6)

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

purge_btn = Button(file_frame, text="purge unsaved files", command=purge_unsaved)
purge_btn.grid(column=4, row=1)

xl_btn = Button(file_frame, text="load xl", command=load_xl)
xl_btn.grid(column=5, row=1)

sxl_btn = Button(file_frame, text="save xl", command=save_just_xl)
sxl_btn.grid(column=6, row=1)

project_name_txt = Entry(file_frame)
project_name_txt.grid(column=2, row=1)
project_name_txt.bind("<Return>", disable_txt)
project_name_txt.bind("<Button 1>", enable_project_name_txt)

next_btn = Button(file_frame, text="next", command=next)
next_btn.grid(column=0, row=4)

save_btn = Button(file_frame, text="save to file ", command=save_to_file)
save_btn.grid(column=1, row=4)

save_toggle_btn = Button(file_frame, text="save/unsave", command=toggle_save)
save_toggle_btn.grid(column=2, row=5)

window.bind("<space>", save_load_next)
window.bind("<Left>", leftKey)
window.bind("<Right>", rightKey)
picture_lbl.bind("<ButtonPress-3>", register_right_click)
picture_lbl.bind("<ButtonRelease-1>", pic_left_mouse)
picture_lbl.bind("<B1-Motion>", left_click)
#picture_lbl.bind("<B3-Motion>", right_click)
picture_lbl.bind("<B3-Motion>", change_angle)
picture_lbl.bind("<ButtonRelease-3>", pic_right_mouse)

picture_lbl.bind("<ButtonPress-2>", rightKey)

window.mainloop()
