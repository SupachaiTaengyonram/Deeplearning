from django.urls import path
from . import views

app_name = "mnistapp"

urlpatterns = [
path("", views.training_page, name="training_page"),
path("train/start", views.start_training, name="start_training"),
path("train/stream", views.train_stream, name="train_stream"),

path("predict", views.predict_digit, name="predict_digit"),
path("draw", views.predict_page, name="predict_page"),
]
