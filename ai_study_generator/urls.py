
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path("admin/", admin.site.urls),
    path('account/',include('account.urls')),
    path('api/',include('summarizing.urls')),
    path('user/',include('user_profile.urls')),

]
