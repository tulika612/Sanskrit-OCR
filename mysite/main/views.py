from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.conf import settings
from django.template import RequestContext

from .forms import SignUpForm, ImageForm
from .models import Image
from .predict import load_model, predict
from .boxes import page_to_line, pdf_to_page, clear_folder

from PIL import Image as Img
from collections import defaultdict
from tesserocr import PyTessBaseAPI, RIL
from collections import OrderedDict
from docx import Document
from docx.enum.text import WD_BREAK

import tensorflow as tf
import pathlib
import json
import numpy as np
import cv2
import os
import glob
from wand.image import Image as wi
import sys
import shutil
import mimetypes

# Create your views here.
def homepage(request):
    global first
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            first=True
            if user is not None:
                login(request, user)
                return redirect('main:success')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request = request,
                    template_name = "main/login.html",
                    context={"form":form})

def success(request):
    global model_detect
    global first
    if request.method == 'POST':
        #lastimage= Image.objects.last()
        #print(lastimage)
        #imagefile= lastimage.imagefile
        print(first)
        if first==True:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess=tf.Session(config=config)
            model_detect = load_model(sess)
            first=False
        print(first)
        form= ImageForm(request.POST, request.FILES )
        #tf.reset_default_graph()
        if form.is_valid():
            form.save()
            lastimage= Image.objects.last()
            imagefile= lastimage.imagefile
            global name
            name = lastimage.filename()
            filetype = form.cleaned_data['Filetype']
            #clear_folder(settings.MEDIA_ROOT+"\\DocFile\\")
            if filetype=='line':
                print(name)
                filetext = 'hello there'
                n=name[:-4]
                filetext=predict(name,model_detect,filetype)
                tmp_path = settings.MEDIA_ROOT+"\\DocFile\\"+n+'.txt'
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(filetext)
                f.close()
                context= {'imagefile': imagefile,'form': form, 'filetext':filetext}
            elif filetype=='page':
                img_path=settings.MEDIA_ROOT+"\\images\\"+name
                filetext = 'hello there'
                page_to_line(img_path)
                #print(name)
                n=name[:-4]
                file_path = settings.MEDIA_ROOT+"\\TextFiles\\"+n+".txt"
                print(file_path)
                full_text = ''
                f=open(file_path,'r')
                for line in f.readlines():
                    filetext=predict(line,model_detect,filetype)
                    full_text=full_text+filetext+'\n'
                tmp_path = settings.MEDIA_ROOT+"\\DocFile\\"+n+'.txt'
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                f.close()
                context= {'imagefile': imagefile,'form': form, 'filetext':full_text}
            elif filetype=='pdf':
                filetext = 'hello there'
                global disp_image
                global final_dic
                global page_list
                img_path=settings.MEDIA_ROOT+"\\images\\"+name
                page_list=pdf_to_page(img_path)
                #print(page_list)
                imagefile=page_list[0]
                disp_image=imagefile
                final_dic = OrderedDict()
                n=name[:-4]
                os.mkdir(settings.MEDIA_ROOT+"\\DocFile\\"+n)
                for pages in page_list:
                    file_name = pages.split("\\")[-1][:-4]
                    file_path = settings.MEDIA_ROOT+"\\TextFiles\\"+file_name+'.txt'
                    doc_file = settings.MEDIA_ROOT+"\\DocFile\\"+n+"\\"+file_name+'.txt'
                    f=open(file_path,'r')
                    full_text = ''
                    for line in f.readlines():
                        filetext=predict(line,model_detect,filetype)
                        full_text=full_text+filetext+'\n'
                    final_dic[pages]=full_text
                    with open(doc_file, "w", encoding="utf-8") as f:
                        f.write(full_text)
                    f.close()
                filetext=final_dic[imagefile]
                context= {'imagefile': imagefile,'form': form, 'filetext':filetext}
            return render(request, 'main/home.html',context)
    else:
        image_form = ImageForm()
        print('in here')

    return render(request, 'main/home.html', {'form': image_form})

def logout_request(request):
    logout(request)
    #messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")

def register(request):
    global first
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            first=True
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('main:success')
    else:
        form = SignUpForm()
    return render(request, 'main/register.html', {'form': form})

def download_file(request):
    global name
    #print(name)
    n=name[:-4]
    if 'pdf' in name:
        print('In pdf')
        doc = Document()
        Path = settings.MEDIA_ROOT+"\\DocFile\\"+n
        shutil.make_archive(settings.MEDIA_ROOT+"\\DocFile\\"+n, 'zip', Path)
        path_to_file=settings.MEDIA_ROOT+"\\DocFile\\"+n+".zip"
        print(path_to_file)
        zip_file = open(path_to_file, 'rb')
        filename=n+'.zip'
        response = HttpResponse(zip_file, content_type='application/force-download')
        #response['Content-Disposition'] = 'attachment; filename="%s"' % 'foo.zip'
    else:
        filename=n+'.txt'
        fl_path = settings.MEDIA_ROOT+"\\DocFile\\"+n+'.txt'
        fl = open(fl_path,"r",encoding="utf-8")
        mime_type, _ = mimetypes.guess_type(fl_path)
        response = HttpResponse(fl, content_type=mime_type)

    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

def next(request):
    if request.method == 'GET':
        global disp_image
        global final_dic
        global page_list
        #print(final_dic)
        try:
            ind = page_list.index(disp_image)
            ind=ind+1
            imagefile = page_list[ind]
        except:
            imagefile = page_list[0]
        disp_image=imagefile
        filetext=final_dic[imagefile]
        context={'imagefile': imagefile,'filetext':filetext}
        return render(request, 'main/home.html',context)


def prev(request):
    if request.method == 'GET':
        global disp_image
        global final_dic
        global page_list
        #print(final_dic)
        try:
            ind = page_list.index(disp_image)
            ind=ind-1
            imagefile = page_list[ind]
        except:
            imagefile = page_list[-1]
        disp_image=imagefile
        filetext=final_dic[imagefile]
        context={'imagefile': imagefile,'filetext':filetext}
        return render(request, 'main/home.html',context)
