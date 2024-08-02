from django.db import models

class User(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    referred_by = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.name
