from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from . import processing
from django.urls import reverse

from skimage import io, filters

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        print(myfile)

        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)

        is_compostable = processing.is_compostable('media/' + filename)
        request.session['is_compostable'] = is_compostable
        #uploaded_file_url = fs.url(filename)
        return HttpResponseRedirect(reverse('results'))
    else:
        return render(request, 'home.html')


def results(request):

    return render(request, 'results.html', context={'is_compostable': request.session['is_compostable']})