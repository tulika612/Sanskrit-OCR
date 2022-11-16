from django.db import models
import os

class Image(models.Model):
    #name= models.CharField(max_length=500)
    #name= models.CharField(max_length=500)
    imagefile= models.FileField(upload_to='images/', null=True, verbose_name="")

    def filename(self):
        return os.path.basename(self.imagefile.name)
