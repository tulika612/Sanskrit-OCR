from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Image


class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False)
    last_name = forms.CharField(max_length=30, required=False)
    email = forms.EmailField(max_length=254)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', )

class ImageForm(forms.ModelForm):
    CHOICES = [('pdf','pdf'),('page','page'),('line','line')]
    Filetype=forms.CharField(label='filetype', widget=forms.RadioSelect(choices=CHOICES))
    class Meta:
        model= Image
        fields= ["Filetype", "imagefile"]
