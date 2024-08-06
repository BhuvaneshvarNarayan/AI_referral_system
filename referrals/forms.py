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
    FormSubmissions = forms.IntegerField(label='Form Submissions', required=True)
    CTR_ProductPage = forms.FloatField(label='CTR Product Page', required=True)
    ResponseTime = forms.FloatField(label='Response Time (hours)', required=True)
    FollowUpEmails = forms.IntegerField(label='Follow Up Emails', required=True)
    SocialMediaEngagement = forms.IntegerField(label='Social Media Engagement', required=True)
    PaymentHistory = forms.IntegerField(label='Payment History', required=True)