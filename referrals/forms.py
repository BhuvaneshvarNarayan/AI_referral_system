from django import forms
from .models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['name', 'email', 'referred_by']

class PredictInterestForm(forms.Form):
    Age = forms.IntegerField(label='Age', required=True)
    Gender = forms.CharField(label='Gender', required=True)
    Location = forms.CharField(label='Location', required=True)
    LeadSource = forms.CharField(label='Lead Source', required=True)
    TimeSpent = forms.IntegerField(label='Time Spent (minutes)', required=True)
    PagesViewed = forms.IntegerField(label='Pages Viewed', required=True)
    EmailSent = forms.IntegerField(label='Email Sent', required=True)
    DeviceType = forms.CharField(label='Device Type', required=True)