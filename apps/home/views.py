# -*- encoding: utf-8 -*-

from multiprocessing import context
from xxlimited import new
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render, redirect
from .models import Todo,Case,Sec,UploadCaseFile
from django.views.decorators.http import require_POST
from .forms import TodoForm,UploadFileForm
import os
from .get_sec_def import getDef
from .getTranslate import getTranslate
from .jpbigru import *
from .similar_cases import *
from .relevant_statues import *

from django.core.files.storage import FileSystemStorage
from django.conf import settings




IMAGE_FILE_TYPES = ['txt']

pred_dict = {0 : 'Rejected'}
pred_dict = {0 : 'Accepted'}

def case_analysis(request):

    form = UploadFileForm(request.POST or None, request.FILES or None)
    if form.is_valid():

        file_content = request.FILES['uploadfile'].read()
        print(file_content)
        file_content = file_content.decode('UTF-8')

        # 1. Judgement Prediction
        pred_dict = judgement_pred_bigru(file_content)
        obj = form.save(commit=False)
        obj.save()
        obj.prediction = pred_dict

        # 2. Similar Cases Retrieval
        similarcases = similarcase(file_content)

        # 3. Law Suggestion
        relevant_statues = similarstat(file_content)

        # 4. Timeline Prediction


        # 5. Translation


        with open(settings.MEDIA_ROOT + '/new_cases/' + obj.uploadfile.url.split('/')[-1], 'r') as f:
            for line in f.readlines():
                obj.uploadfile_description += line.strip()

        obj.save()

        context = {'form' : form, 'similarcases': similarcases}
        html_template = loader.get_template('home/case_analysis.html')
        return HttpResponse(html_template.render(context, request))

    context = {'form' : form}
    html_template = loader.get_template('home/case_analysis.html')
    return HttpResponse(html_template.render(context, request))


def uploaded_cases(request):
    files = UploadCaseFile.objects.all()
    for f in files:
        f.uploadfile_description = f.uploadfile_description[0:70] + '...'

    context = { 'files': files}
    html_template = loader.get_template('home/uploaded_cases.html')
    return HttpResponse(html_template.render(context, request))
   

use_model = True
@require_POST
@login_required(login_url="/login/")
def get_similar_cases(request, id=None):
    case = Case.objects.get(id=id)
    similarcases = similarcase(case.case_description)
    # case.similarcases = similarcases
    # case.save()
    

    context = {'case' : case, 'similarcases': similarcases}
    html_template = loader.get_template('home/similar_cases.html')
    return HttpResponse(html_template.render(context, request))




def translate(request):
    
    form = UploadFileForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            file_content = request.FILES['uploadfile'].read()
            file_content = file_content.decode('UTF-8')  

            language = request.POST.get("dropdown", "")
            print("Language:", language)
            translated = file_content

            context = {'language': language,'before_trans': file_content, 'translated': translated, 'form': form}
        
        html_template = loader.get_template('home/translate.html')
        return HttpResponse(html_template.render(context, request))

    context = {'form':form}
    html_template = loader.get_template('home/translate.html')
    return HttpResponse(html_template.render(context, request))


def predict_judgement(request):    
    form = UploadFileForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            file_content = request.FILES['uploadfile'].read()
            file_content = file_content.decode('UTF-8')
            prediction = judgement_pred_bigru(file_content)

            context = {'prediction': prediction , 'form': form}
        
        html_template = loader.get_template('home/translate.html')
        return HttpResponse(html_template.render(context, request))

    context = {'form':form}
    html_template = loader.get_template('home/translate.html')
    return HttpResponse(html_template.render(context, request))



    # #with open(newfile.uploadfile.path,'r') as f:
    # #    for line in f.readlines():
    # #        case_content += line.strip()
    # form=UploadFileForm()

    # case_content = ''

    # language=request.POST.get("dropdown", "")
    # print(language)
    # output=getTranslate(language,case_content)

    # context={'output':output,'language':language,'form':form}

    # html_template = loader.get_template('home/translator.html')
    # return HttpResponse(html_template.render(context, request))




@require_POST
@login_required(login_url="/login/")

def sec(request):
    
    input=request.POST.get("SecNo", "")
    output=getDef(int(input))

    sec_def=Sec(sec_name=input,sec_def=output)
    sec_def.save()
    context={'output':output}

    html_template = loader.get_template('home/sec_def.html')
    return HttpResponse(html_template.render(context, request))
                                                

data_path = "apps\sihdoc\Object_casedocs"

@require_POST
@login_required(login_url="/login/")
def addCasetoDB(request):
    context = {}

    for case_path in os.listdir(data_path):
        
        case_path_new = data_path + '/' + case_path        
        # read case
        case_filename = case_path
        content = ""
        if ".txt" in case_filename:
            with open(case_path_new, 'r') as f:
                for line in f.readlines():
                    content += line.strip()
                
            case_description = content
            case_status = "completed"

            new_case = Case(case_name=case_filename, case_description=case_description, case_status=case_status)
            new_case.save()

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))

def analysis(request):
    context = {}

    html_template = loader.get_template('home/analysis.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def index(request):
    form=TodoForm()
    todo_list=Todo.objects.order_by('id')
    context = {'segment': 'index','todo_list' : todo_list ,'form' : form}


    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
