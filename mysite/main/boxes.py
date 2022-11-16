from PIL import Image as Img
from collections import defaultdict
from tesserocr import PyTessBaseAPI, RIL
import pathlib
import json
import numpy as np
import cv2
import os
import glob
from wand.image import Image as wi
import sys
import shutil

from django.conf import settings

def page_to_line(img_path):
    print('page_to_line')
    allboxes = segment_images(img_path)
    bbx_json = convert_to_bbx(allboxes)
    #clear_folder(settings.MEDIA_ROOT+"\\Bbx_file\\")
    #clear_folder(settings.MEDIA_ROOT+"\\LineImages\\")
    #clear_folder(settings.MEDIA_ROOT+"\\TextFiles\\")
    with open(settings.MEDIA_ROOT+"\\Bbx_file\\boxes.bbx", 'w') as fp:
        json.dump(bbx_json, fp)
    create_images(bbx_json,img_path,'page')



def create_images(data_in,folder_path,type):
        for i in data_in["images"].keys():
            if type=='page':
                img_path=folder_path
            else:
                img_path=folder_path+i
            print(img_path)
            thresh1= cv2.imread(img_path,0)
            # ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
            height, width = thresh1.shape[:2]
            height, width = thresh1.shape[:2]
            f=1
            img_name= i[:-4]
            print(img_name)
            new_path=settings.MEDIA_ROOT+"\\LineImages\\" + img_name

            print(new_path)
            os.mkdir(new_path)
            #clear_folder(new_path)
            for annot in data_in["images"][i]["annotations"]:
                xmin=int(annot['bbox']["xmin"]*width)
                xmax=int(annot['bbox']["xmax"]*width)
                ymin=int(annot['bbox']["ymin"]*height)
                ymax=int(annot['bbox']["ymax"]*height)
                new_name = new_path + "\\" + img_name +"_" + str(f)

                cv2.imwrite(new_name+".jpg",thresh1[ymin:ymax,xmin:xmax])
                txt_file_name=settings.MEDIA_ROOT+"\\TextFiles\\"+img_name+ ".txt"
                file_object = open(txt_file_name, 'a')
                file_object.write(new_name+ ".jpg \n")
                file_object.close()
                f=f+1


def segment_images2(img_paths, save_lines_as_images=False, save_coords = False):
    allboxes = defaultdict(list)

    if save_coords is True:
        pathlib.Path("Results/Coords").mkdir(parents=True, exist_ok=True)

    with PyTessBaseAPI(lang = 'Devanagari',path='C:\\Users\\tulik\\anaconda3\\envs\\venv\\Lib\\site-packages\\tessdata') as api:
        for page_no, path2 in enumerate(img_paths):
            if path2.endswith('.DS_Store'):
                continue
            if path2.endswith('.bbx'):
                continue
            if os.path.isdir(path2):
                continue


            if save_lines_as_images is True:
                pathlib.Path(settings.MEDIA_ROOT+"\\Lineimages\\" + str(page_no +1)).mkdir(parents=True, exist_ok=True)
            print("hi")
            image = Img.open(path2)
            img = np.array(image)
            image = image.convert('L')
            api.SetImage(image)
            boxes = api.GetComponentImages(RIL.TEXTLINE, False)

            # print(boxes)
            print('Found {} textline image components.'.format(len(boxes)))

            boxlist = []

            for (im, box, _, _) in boxes:
                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                x,y,w,h = box['x'], box['y'], box['w'], box['h']

                boxlist.append([x,y,w,h])
                # api.SetRectangle(x, y, w, h)
                # ocrResult = api.GetUTF8Text()
                # conf = api.MeanTextConf()
                # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                # "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

            # Merge Boxes on same line

            boxlist = merge_boxes(boxlist)

            coord_dict = {}
            img2 = img.copy()

            for i, (left, top, right, bottom) in enumerate(boxlist):

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, str(i), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                img_name = path2.split("\\")[-1]
                xmin = left/img.shape[1]
                xmax = right/img.shape[1]
                ymin = top/img.shape[0]
                ymax = bottom/img.shape[0]
                allboxes[img_name].append((xmin, xmax, ymin, ymax))

                # if save_lines_as_images is True:
                #     line_img = img2[top:bottom, left:right, :]
                #     filename = 'media/LineImages/Page' + str(page_no +1) +"/" + str(i+1) + ".jpg"
                #     cv2.imwrite(filename, line_img)

                if save_coords is True:
                    box_dict = {"top_left": [left, top],
                        "top_right":[right, top],
                        "bottom_left":[left, bottom],
                        "bottom_right": [right, bottom]
                        }

                    coord_dict["box"+str(i+1)] = box_dict


            if save_coords is True:
                filename = 'Results/Coords/Page' + str(page_no + 1) + '.json'
                with open(filename, 'w') as outfile:
                    json.dump(coord_dict, outfile)

    return allboxes


def segment_images(img_paths, show_segmented_images=True, save_segmented_images=False, save_lines_as_images=False, save_coords = False):
        allboxes = defaultdict(list)
        print('segment_images')
        if save_segmented_images is True:
            pathlib.Path("Results/Pages").mkdir(parents=True, exist_ok=True)

        if save_coords is True:
            pathlib.Path("Results/Coords").mkdir(parents=True, exist_ok=True)

        with PyTessBaseAPI(lang = 'Devanagari',path='C:\\Users\\tulik\\anaconda3\\envs\\venv\\Lib\\site-packages\\tessdata') as api:
            page_no=0
            path2=img_paths
            if 1:
                # if path2.endswith('.DS_Store'):
                #     continue
                # if path2.endswith('.bbx'):
                #     continue
                # if os.path.isdir(path2):
                #     continue

                if save_lines_as_images is True:
                    pathlib.Path("Results/Lines/Page" + str(page_no +1)).mkdir(parents=True, exist_ok=True)

                image = Img.open(path2)
                img = np.array(image)
                image = image.convert('L')
                api.SetImage(image)
                boxes = api.GetComponentImages(RIL.TEXTLINE, False)

                # print(boxes)
                print('Found {} textline image components.'.format(len(boxes)))

                boxlist = []

                for (im, box, _, _) in boxes:
                    # im is a PIL image object
                    # box is a dict with x, y, w and h keys
                    x,y,w,h = box['x'], box['y'], box['w'], box['h']

                    boxlist.append([x,y,w,h])
                    # api.SetRectangle(x, y, w, h)
                    # ocrResult = api.GetUTF8Text()
                    # conf = api.MeanTextConf()
                    # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                    # "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

                # Merge Boxes on same line

                boxlist = merge_boxes(boxlist)

                coord_dict = {}
                img2 = img.copy()

                for i, (left, top, right, bottom) in enumerate(boxlist):

                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(img, str(i), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    img_name = path2.split("\\")[-1]
                    xmin = left/img.shape[1]
                    xmax = right/img.shape[1]
                    ymin = top/img.shape[0]
                    ymax = bottom/img.shape[0]
                    allboxes[img_name].append((xmin, xmax, ymin, ymax))

                    if save_lines_as_images is True:
                        line_img = img2[top:bottom, left:right, :]
                        filename = 'Results/Lines/Page' + str(page_no +1) +"\\" + str(i+1) + ".jpg"
                        cv2.imwrite(filename, line_img)

                    if save_coords is True:
                        box_dict = {"top_left": [left, top],
                            "top_right":[right, top],
                            "bottom_left":[left, bottom],
                            "bottom_right": [right, bottom]
                            }

                        coord_dict["box"+str(i+1)] = box_dict

                if show_segmented_images is True:
                    scale_percent = 30 # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # resize image
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    #cv2_imshow(resized)

                if save_segmented_images is True:
                    filename = "Results/Pages/" + str(page_no +1) + ".jpg"
                    cv2.imwrite(filename, img)

                if save_coords is True:
                    filename = 'Results/Coords/Page' + str(page_no + 1) + '.json'
                    with open(filename, 'w') as outfile:
                        json.dump(coord_dict, outfile)

        return allboxes

def merge_boxes(boxlist):
    boxlist_np = np.asarray(boxlist)
    print('merge_boxes')
    boxlist_np = boxlist_np[boxlist_np[:,1].argsort()] # sort by y
    x,y,w,h = boxlist_np.T
    boxlist = np.column_stack((x,y,x+w,y+h)).tolist()
    median_height = np.median(h) # We filter by half of median height.

    diff = abs(np.ediff1d(y)) # Difference of consecutive elements
    diff = (diff <= (median_height/2)).nonzero()[0] # Get indices of elements where difference less than avg height/2
    for i in diff[::-1]: # Go in reverse so that indices are not changed by popping.
        left = min(x[i], x[i+1])
        top = min(y[i], y[i+1])
        right = max(x[i] + w[i], x[i+1] + w[i+1])
        bottom = max(y[i] + h[i], y[i+1] + h[i+1])

        boxlist[i] = (left,top,right,bottom)
        boxlist.pop(i+1)

    return boxlist

def convert_to_bbx(allboxes):

    final_json = {}

    final_json["mask"] = None
    final_json["mask_name"] = ""
    final_json["images"] = {}

    for img_name, boxes in allboxes.items():
        page_dict = {}
        page_dict["attribution"] = ""
        page_dict["license"] = ""
        page_dict["license_url"] = ""
        annotations = []
        for box in boxes:
            box_dict = {}
            box_dict["created_by"] = "human"
            box_dict["updated_by"] = "human"
            box_dict["bbox"] = {
                "xmin" : box[0],
                "xmax" : box[1],
                "ymin" : box[2],
                "ymax" : box[3],
            }
            box_dict["label"] = "N/A"
            box_dict["occluded"] = "N"
            box_dict["truncated"] = "N"
            box_dict["difficult"] = "N"
            box_dict["schema"] = "1.0.0"

            annotations.append(box_dict)

        page_dict["annotations"] = annotations
        final_json["images"][img_name] = page_dict

    final_json["analysts"] = []
    final_json["schema"] = "1.0.0"

    return final_json

def pdf_to_page(pdf_path):
    #clear_folder(settings.MEDIA_ROOT+"\\PageImages\\")
    #clear_folder(settings.MEDIA_ROOT+"\\TextFiles\\")
    #clear_folder(settings.MEDIA_ROOT+"\\LineImages\\")
    #clear_folder(settings.MEDIA_ROOT+"\\Bbx_file\\")
    print(pdf_path)
    page_list=[]
    pdf = wi(filename=pdf_path, resolution=100)
    pdf_name=pdf_path.split("\\")[-1]
    pdfImg = pdf.convert('jpg')
    j = 1
    imgBlobs = []
    img1= []
    new_path=settings.MEDIA_ROOT+"\\PageImages\\"

    if(os.path.isdir(new_path)):
        files = glob.glob(new_path, recursive=True)
        for file in files:
            try:
                shutil.rmtree(file)
            except OSError as e:
                print(e)
    os.mkdir(new_path)
    for img in pdfImg.sequence:
        page = wi(image=img)
        filename=settings.MEDIA_ROOT+"\\PageImages\\"+pdf_name+"-Page"+str(j)+".jpg"
        page_list.append(filename)
        page.save(filename=filename)
        img1.append(cv2.imread(str(j)+".jpg"))
        j += 1

    #creating line images from pages
    images_path = [new_path + s for s in sorted(os.listdir(new_path))]
    print(images_path)
    allboxes = segment_images2(images_path)
    bbx_json = convert_to_bbx(allboxes)
    with open(settings.MEDIA_ROOT+'\\Bbx_file\\'+pdf_name+'.bbx', 'w') as fp:
        json.dump(bbx_json, fp)
    create_images(bbx_json,new_path,'pdf')


    first_page=settings.MEDIA_ROOT+"\\PageImages\\"+pdf_name+"-Page1.jpg"

    return page_list

def clear_folder(new_path):
        if(os.path.isdir(new_path)):
            files = glob.glob(new_path, recursive=True)
            for file in files:
                try:
                    shutil.rmtree(file)
                except OSError as e:
                    print(e)
        os.mkdir(new_path)
