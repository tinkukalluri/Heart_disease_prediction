from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response  
from django.http import HttpResponse,JsonResponse

from django.views.decorators.csrf import csrf_exempt
# Create your views here.



def index(request):
    return render(request, "index.html")
    # return HttpResponse("tinku")
