# Generated by Django 3.1.4 on 2020-12-01 18:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_image_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='image',
            name='name',
        ),
    ]