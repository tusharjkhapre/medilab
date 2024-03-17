from operator import mod
from pyexpat import model
from re import M
from django.db import models
from django.contrib.auth.models import User

# Create your models here.
DOCTOR_STATUS = ((1, 'Authorize'), (2, 'UnAuthorize'))

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    contact = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    dob = models.DateField(null=True)
    image = models.FileField(null=True)

    def __str__(self):
        return self.user.username

class Doctor(models.Model):
    status = models.IntegerField(DOCTOR_STATUS, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    contact = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    category = models.CharField(max_length=100, null=True)
    doj = models.DateField(null=True)
    dob = models.DateField(null=True)
    image = models.FileField(null=True)

    def __str__(self):
        return self.user.username

class Admin_Helath_CSV(models.Model):
    name = models.CharField(max_length=100, null=True)
    csv_file = models.FileField(null=True, blank=True)

    def __str__(self):
        return self.name

class Search_Data(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, null=True)
    prediction_accuracy = models.CharField(max_length=100,null=True,blank=True)
    result = models.CharField(max_length=100,null=True,blank=True)
    values_list = models.CharField(max_length=100,null=True,blank=True)
    predict_for = models.CharField(max_length=100,null=True,blank=True)
    uploaded  = models.FileField(null=True, blank=True)
    output  = models.FileField(null=True, blank=True)
    created = models.DateTimeField(auto_now=True,null=True)

    def __str__(self):
        return self.patient.user.username

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    messages = models.TextField(null=True)
    date = models.DateField(auto_now=True)

    def __str__(self):
        return self.user.user.username

class GeneralHealthProblem(models.Model):
    name = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.name

class Blood_Donation(models.Model):
    status = models.CharField(max_length=100, null=True, blank=True)
    user = models.ForeignKey(Patient, on_delete=models.CASCADE, null=True, blank=True)
    blood_group = models.CharField(max_length=100, null=True, blank=True)
    place = models.CharField(max_length=100, null=True, blank=True)
    purpose = models.CharField(max_length=100, null=True, blank=True)
    created = models.DateTimeField(auto_now=True,null=True)
    active = models.BooleanField(null=True, blank=True, default=False)

    def __str__(self):
        return self.user.user.username
