from django.urls import path, include, re_path
from . import views

app_name = "main"

urlpatterns = [
    path("",views.homepage, name= "homepage"),
    path('register/',views.register, name= "register"),
    path('login/',views.homepage, name= "register"),
    path('home/',views.success, name= "success"),
    path("logout/", views.logout_request, name="logout"),
    path("home/next", views.next),
    path("home/prev", views.prev),
    path("home/download", views.download_file),
    path("",views.homepage )
]
