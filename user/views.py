from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
# models
from . import models

# @csrf_exempt
def login_view(request):
    if request.method=='POST':
        post_data=json.loads(request.body)
        print(type(post_data))
        # we r using request.body because we r not using <form/>
        # f you need to access raw or non-form data posted in the request, access this through the HttpRequest.body attribute instead.
        username=post_data.get('username')
        password=post_data.get('password')
        print(username, password)
        queryset=models.Users.objects.filter(username=username, password=password)
        if queryset.exists():
            user=queryset[0]
            if not request.session.exists(request.session.session_key):
                request.session.create()
                request.session["key"]=request.session.session_key
                request.session["user_id"]=user.id
                request.session["username"]=username
            return JsonResponse({
                'id':user.id,
                'username':username
            })
        else:
            return JsonResponse({
                "error":"no User"
            })
    return JsonResponse({
        "error":"nope"
    })


def authenticate_view(request):
    if request.method=="GET":
        print("authenticate" , request.session.exists(request.session.session_key))
        if request.session.exists(request.session.session_key):
            return JsonResponse({
                'id':request.session.get('user_id'),
                "username": request.session.get('username')
            })
        else:
            return JsonResponse({
                "error":"please login"
            })
    return HttpResponse('from user authenticate')


def logout(request):
    if not request.session.exists(request.session.session_key):
        return JsonResponse({"status": True})
    else:
        # del request.session['user_id']
        # del request.session["username"]
        request.session.flush()
        print("deleted")
        return JsonResponse({"status": True})