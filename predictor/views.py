from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.contrib import messages
import warnings

from .forms import DocumentForm
from .models import Document
from .Metadata import getmetadata
from .predict import predict_gen

warnings.simplefilter('ignore')


class IndexView(ListView):
    template_name = 'music/index.html'

    def get_queryset(self):
        return True


def model_form_upload(request):
    documents = Document.objects.all()

    if request.method == 'POST':
        if len(request.FILES) == 0:
            messages.error(request, 'Upload a file')
            return redirect("predictor:index")

        form = DocumentForm(request.POST, request.FILES)

        if form.is_valid():
            uploadfile = request.FILES['document']

            if not uploadfile.name.lower().endswith('.wav'):
                messages.error(request, 'Only .wav file type is allowed')
                return redirect("predictor:index")

            try:
                uploadfile.seek(0)   # important for uploaded file objects
                meta = getmetadata(uploadfile)
                genre = predict_gen(meta)

                context = {'genre': genre}
                return render(request, 'music/result.html', context)

            except Exception as e:
                messages.error(request, f'Prediction failed: {str(e)}')
                return redirect("predictor:index")

        else:
            messages.error(request, 'Invalid form submission')
            return redirect("predictor:index")

    else:
        form = DocumentForm()

    return render(request, 'music/result.html', {
        'documents': documents,
        'form': form
    })