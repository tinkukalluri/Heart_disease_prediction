from django.shortcuts import render

def index(request):
    return render(request,'index.html',{})


def logout(request):
    request.session.flush()
    print("deleted")
    return render(request,'index.html',{})