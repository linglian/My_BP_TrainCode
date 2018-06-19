"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import include, url
from django.contrib import admin
from . import get
from django.views.static import serve

urlpatterns = [
    url('admin/', admin.site.urls),
    url('get/', get.get),
    url('list/', get.list),
    url('shaixuan/', get.getButton),
    url('index/', get.getIndex),
    url('dingdan/', get.dingdan),
    url('gwc/', get.gwc),
    url('sjd_list', get.sjd_get),
    url('sjd_remove', get.sjd_remove),
    url('sjd_add_cloth', get.sjd_add_cloth),
    url('get_all_cloth', get.get_all_cloth),
    url('create_sjd', get.create_sjd),
    url('save_sjd', get.save_sjd),
    url('del_sjd', get.del_sjd),
    url('upload', get.uploadImg),
    url(r'^medias/(?P<path>.*)$', serve, {'document_root': '/media/lee/data/macropic/newp/new_数据加强版_整理版/'}), 
]
