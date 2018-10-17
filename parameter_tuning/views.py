from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.shortcuts import redirect
from django.shortcuts import render, render_to_response
from django.core.files.storage import FileSystemStorage
import os
from django.contrib import messages
from deep_learning import settings
import zipfile
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from parameter_tuning import views
from django.http import HttpResponse
from keras import backend
from wsgiref.util import FileWrapper
import mimetypes
from django.utils.encoding import smart_str
import math
import smtplib
from email.mime.text import MIMEText

views.classifier = Sequential()
views.file_flag = False


@csrf_exempt
def logout(request):
    del request.session['username']
    return render(request,'index.html',{})

@csrf_exempt
def contact(request):
    return render(request,'contact.html',{})

@csrf_exempt
def send_mail(request):
    fromaddr = 'deep.learning.framework@gmail.com'
    toaddrs  = 'sivaanandhmuneeswaran_cs@mepcoeng.ac.in'
    username = 'deep.learning.framework@gmail.com'
    password = 'Bio5283#'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    name = request.POST.get('name')
    mail = request.POST.get('email')
    message = request.POST.get('message')
    msg = """
    You have received a new message from your website contact form.
    Here are the details:
    Name   : %s
    Email  : %s
    Message: %s"""%(name,mail,message)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()
    return render(request, 'index.html', {'mail_flag': True})

# Create your views here.
@csrf_exempt
def index(request):
    if(request.method == "POST"):
        if(request.session.has_key('username')):
            path = os.path.join(settings.BASE_DIR,'media')
            name = request.session['username']
            views.email = request.session['username']
            path = f'{path}/{name}'
            if not os.path.exists(path):
                os.makedirs(path)
            if('new_upload' in request.POST):
                return render(request,'upload.html',{})
            if('old_upload' in request.POST):
                dir_list = os.listdir(path)
                if(dir_list):
                    return render(request,'select_dataset.html',{'dir_list':dir_list})
                else:
                    return render(request,'index.html',{'no_dataset':True})
        else:
            views.email = request.POST.get('email')
            request.session['username']= views.email
            path = os.path.join(settings.BASE_DIR,'media')
            name = request.session['username']
            path = f'{path}/{name}'
            if not os.path.exists(path):
                os.makedirs(path)
            if('new_upload' in request.POST):
                return render(request,'upload.html',{})
            if('old_upload' in request.POST):
                dir_list = os.listdir(path)
                if(dir_list):
                    return render(request,'select_dataset.html',{'dir_list':dir_list})
                else:
                    return render(request,'index.html',{'no_dataset':True})
    else:
        return render(request,'index.html',{})


@csrf_exempt
def upload(request):
    if(request.method == "POST"):
        if('upload_btn' in request.POST):
            views.email = request.session['username']
            myfile = request.FILES['my_file']
            if(myfile):
                fs = FileSystemStorage()
                views.folder_name = myfile.name.split(".")[0]
                path = os.path.join(settings.BASE_DIR,'media')
                path = f'{path}/{views.email}'
                fs = FileSystemStorage(location=f'{path}')
                filename = fs.save(myfile.name, myfile)
                PROJECT_ROOT = os.path.join(settings.BASE_DIR,'media')
                filename = os.path.join(PROJECT_ROOT,f'{views.email}/{filename}')
                unzip(filename,PROJECT_ROOT)
                views.file_flag = True
                return render(request,'upload.html',{'flag': True})
            else:
                return render(request,'upload.html',{'no_file_flag': True})
        if('train_model_btn' in request.POST):
            if(views.file_flag):
                views.classifier = Sequential()
                views.progress = 0
                views.progress_flag = True
                return render(request,'analyse.html',{})
            else:
                return render(request,'upload.html',{'not_flag':True})
    else:
        return render(request,'upload.html',{})


def unzip(filename,PROJECT_ROOT):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(f'{PROJECT_ROOT}/{views.email}')
    zip_ref.close()
    if(os.path.exists(filename)):
        os.remove(filename)

@csrf_exempt
def use_existing_dataset(request):
    views.classifier = Sequential()
    views.progress = 0
    views.progress_flag = True
    dataset = request.POST.get('existing_dataset')
    views.folder_name = dataset
    return render(request,'analyse.html',{})


@csrf_exempt
def add_conv_layer(request):
    filter_size = request.POST.get('filter_size')
    conv_height = request.POST.get('conv_height')
    conv_width = request.POST.get('conv_width')
    views.img_height = request.POST.get('img_height')
    views.img_width = request.POST.get('img_width')
    channel = request.POST.get('channel')
    activation_function_conv = request.POST.get('activation_function_conv')
    pool_height = request.POST.get('pool_height')
    pool_width = request.POST.get('pool_width')
    with backend.get_session().graph.as_default() as g:
        views.classifier.add(Conv2D(int(filter_size), (int(conv_height), int(conv_width)), input_shape = (int(views.img_height), int(views.img_width), int(channel)), activation = activation_function_conv))
        views.classifier.add(MaxPooling2D(pool_size = (int(pool_height), int(pool_width))))
    if('add_layer_btn' in request.POST):
        return render(request,'analyse_new.html',{})
    else:
        return render(request,'train.html',{})

@csrf_exempt
def download(request):
    with backend.get_session().graph.as_default() as g:
        path = os.path.join(settings.BASE_DIR,f'media/{views.email}/{views.folder_name}')
        views.classifier.save_weights(f'{path}/weights.h5')
        views.classifier.save(f'{path}/model.h5')
    return render(request,'result.html',{})

@csrf_exempt
def download_model(request):
    file_name = 'model.h5'
    file_path = settings.MEDIA_ROOT +'/'+ views.email + '/' + views.folder_name + '/' + file_name
    file_wrapper = FileWrapper(open(file_path,'rb'))
    file_mimetype = mimetypes.guess_type(file_path)
    response = HttpResponse(file_wrapper, content_type=file_mimetype )
    response['X-Sendfile'] = file_path
    response['Content-Length'] = os.stat(file_path).st_size
    response['Content-Disposition'] = 'attachment; filename=%s' % smart_str(file_name)
    return response

@csrf_exempt
def download_weight(request):
    file_name = 'weights.h5'
    file_path = settings.MEDIA_ROOT +'/' + views.email + '/' + views.folder_name + '/' + file_name
    file_wrapper = FileWrapper(open(file_path,'rb'))
    file_mimetype = mimetypes.guess_type(file_path)
    response = HttpResponse(file_wrapper, content_type=file_mimetype )
    response['X-Sendfile'] = file_path
    response['Content-Length'] = os.stat(file_path).st_size
    response['Content-Disposition'] = 'attachment; filename=%s' % smart_str(file_name) 
    return response

@csrf_exempt
def add_conv_layer_new(request):
    filter_size = request.POST.get('filter_size')
    conv_height = request.POST.get('conv_height')
    conv_width = request.POST.get('conv_width')
    activation_function_conv = request.POST.get('activation_function_conv')
    pool_height = request.POST.get('pool_height')
    pool_width = request.POST.get('pool_width')
    with backend.get_session().graph.as_default() as g:
        views.classifier.add(Conv2D(int(filter_size), (int(conv_height), int(conv_width)), activation = activation_function_conv))
        views.classifier.add(MaxPooling2D(pool_size = (int(pool_height), int(pool_width))))
    if('add_layer_btn_new' in request.POST):
        return render(request,'analyse_new.html',{})
    else:
        return render(request,'train.html',{})

@csrf_exempt
def train_model(request):
    class_count = request.POST.get('class_count')
    activation_function_fc = request.POST.get('activation_function_fc')
    optimizers = request.POST.get('optimizer')
    loss_function = request.POST.get('loss_function')
    shear_range = request.POST.get('shear_range')
    zoom_range = request.POST.get('zoom_range')
    batch_size = request.POST.get('batch_size')
    views.epoch_count = request.POST.get('epoch_count')
    horizontal_flip = True
    vertical_flip = True
    path = os.path.join(settings.BASE_DIR,f'media/{views.email}/{views.folder_name}')
    with backend.get_session().graph.as_default() as g:
        views.classifier.add(Flatten())
        views.classifier.add(Dense(units = int(class_count), activation = activation_function_fc))
        views.classifier.compile(optimizer = optimizers, loss = loss_function, metrics = ['accuracy'])

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = float(shear_range),
                                        zoom_range = float(zoom_range),
                                        horizontal_flip = True,
                                        vertical_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        views.training_set = train_datagen.flow_from_directory(f'{path}/train',
                                                        target_size = (int(views.img_height), int(views.img_width)),
                                                        batch_size =int(batch_size))

        views.test_set = test_datagen.flow_from_directory(f'{path}/test',
                                                    target_size = (int(views.img_height), int(views.img_width)),
                                                    batch_size = int(batch_size))
        views.steps_per_epoch = math.ceil((len(views.training_set.filenames))/(int(batch_size)))
        views.validation_steps = math.ceil((len(views.test_set.filenames))/(int(batch_size)))
        history = views.classifier.fit_generator(views.training_set,steps_per_epoch = views.steps_per_epoch,epochs = 1,validation_data = views.test_set,validation_steps = views.validation_steps)
        train_accuracy = round(history.history['acc'][0],3)
        train_loss = round(history.history['loss'][0],3)
        validation_accuracy = round(history.history['val_acc'][0],3)
        validation_loss = round(history.history['val_loss'][0],3)

    views.increase = 100/int(views.epoch_count)
    views.progress = views.progress + views.increase
    views.progress_flag = True
    if(views.progress == 100):
        views.progress_flag = False
    return render(request,'training.html',{'increase':views.progress,'progress_flag':views.progress_flag,'train_accuracy':train_accuracy,'train_loss':train_loss,'validation_accuracy':validation_accuracy,'validation_loss':validation_loss})

@csrf_exempt
def training_model(request):
    with backend.get_session().graph.as_default() as g:
        history = views.classifier.fit_generator(views.training_set,steps_per_epoch = views.steps_per_epoch,epochs = 1,validation_data = views.test_set,validation_steps = views.validation_steps)
        train_accuracy = round(history.history['acc'][0],3)
        train_loss = round(history.history['loss'][0],3)
        validation_accuracy = round(history.history['val_acc'][0],3)
        validation_loss = round(history.history['val_loss'][0],3)
        views.progress = views.progress + views.increase
        views.progress_flag = True
        if(views.progress == 100):
            views.progress_flag = False
    return render(request,'training.html',{'increase':views.progress, 'progress_flag':views.progress_flag,'train_accuracy':train_accuracy,'train_loss':train_loss,'validation_accuracy':validation_accuracy,'validation_loss':validation_loss})