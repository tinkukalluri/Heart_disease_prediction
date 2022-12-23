from django.db import models

# Create your models here.



class Users(models.Model):
    username = models.CharField(max_length=255 , unique=True , null=False)
    password = models.CharField(max_length=255 , unique=False, null=False)