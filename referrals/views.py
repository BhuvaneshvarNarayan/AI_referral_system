from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import User
from .ml_model import predict
from .forms import UserForm


@csrf_exempt
def add_user(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'referrals/add_user.html', {'message': 'User added successfully'})
    else:
        form = UserForm()
    return render(request, 'referrals/add_user.html', {'form': form})


def get_users(request):
    users = User.objects.all()
    user_list = [{'id': user.id, 'name': user.name, 'email': user.email, 'referred_by': user.referred_by} for user in
                 users]
    return JsonResponse(user_list, safe=False)


@csrf_exempt
def predict_interest(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        prediction = predict(data)
        return JsonResponse({'interested': prediction})
    else:
        if request.method == 'GET':
            return render(request, 'referrals/predict_interest_form.html')
        return JsonResponse({'error': 'POST request required'}, status=400)


def predict_interest_form(request):
    return render(request, 'referrals/predict_interest_form.html')


def predict_interest_result(request):
    interested = request.GET.get('interested', 'N/A')
    return render(request, 'referrals/predict_interest_result.html', {'interested': interested})
