from django.urls import path # type: ignore
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('register/', views.register, name='register'),
    path('start/', views.start_attendance, name='start_attendance'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('notfound/', views.not_found, name='not_found'),
    path('download-excel/', views.download_attendance_summary, name='download_excel'),
    #path('confirm_attendance/', views.confirm_attendance, name='confirm_attendance'),
]
