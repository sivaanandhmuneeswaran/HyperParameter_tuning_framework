from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('upload',views.upload,name='upload'),
    path('add_conv_layer',views.add_conv_layer,name='add_conv_layer'),
    path('add_conv_layer_new',views.add_conv_layer_new,name='add_conv_layer_new'),
    path('train_model',views.train_model,name='train_model'),
    path('training_model',views.training_model,name='training_model'),
    path('download',views.download,name='download'),
    path('download_model',views.download_model,name='download_model'),
    path('download_weight',views.download_weight,name='download_weight'),
    path('contact',views.contact,name='contact'),
    path('send_mail',views.send_mail,name='send_mail'),
    path('logout',views.logout,name='logout'),
    path('use_existing_dataset',views.use_existing_dataset,name='use_existing_dataset'),
]