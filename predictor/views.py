import os
import base64
import warnings
from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.contrib import messages

from .forms import DocumentForm
from .Metadata import getmetadata, generate_visualizations
from .predict import predict_gen

warnings.simplefilter('ignore')

ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.au', '.aiff')


class IndexView(ListView):
    template_name = 'music/index.html'

    def get_queryset(self):
        return True


def model_form_upload(request):
    if request.method == 'POST':
        if len(request.FILES) == 0:
            messages.error(request, 'Please select or drag & drop an audio file to classify.')
            return redirect("predictor:index")

        form = DocumentForm(request.POST, request.FILES)

        if form.is_valid():
            # Retrieve uploaded audio file
            uploadfile = request.FILES.get('document') or request.FILES.get('file')
            if not uploadfile:
                messages.error(request, 'No audio file found in upload request.')
                return redirect("predictor:index")

            # Validate file extension
            _, ext = os.path.splitext(uploadfile.name)
            if ext.lower() not in ALLOWED_EXTENSIONS:
                allowed_str = ", ".join(ALLOWED_EXTENSIONS)
                messages.error(
                    request,
                    f'Unsupported file format "{ext}". Supported formats are: {allowed_str}'
                )
                return redirect("predictor:index")

            # Selected classifier model: 'indian' (default) or 'global'
            model_type = request.POST.get('model_type', 'indian').lower()
            if model_type not in ('indian', 'global'):
                model_type = 'indian'

            # Selected architecture: 'svm' (Classical ML) or 'cnn' (Deep Learning)
            architecture = request.POST.get('architecture', 'svm').lower()
            if architecture not in ('svm', 'cnn'):
                architecture = 'svm'

            try:
                # In-memory feature extraction and audio loading
                uploadfile.seek(0)
                meta, y, sr = getmetadata(uploadfile, return_audio=True)

                # Compute model inference and class confidence distributions
                prediction_result = predict_gen(
                    meta,
                    model_type=model_type,
                    architecture=architecture,
                    y=y,
                    sr=sr
                )

                # Generate waveform and mel-spectrogram visualizations
                waveform_b64, spectrogram_b64 = generate_visualizations(y, sr)

                # Generate base64 audio URI for in-browser playback
                uploadfile.seek(0)
                audio_b64 = base64.b64encode(uploadfile.read()).decode('utf-8')
                mime_map = {
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.ogg': 'audio/ogg',
                    '.opus': 'audio/ogg',
                    '.flac': 'audio/flac',
                    '.m4a': 'audio/mp4',
                    '.aac': 'audio/aac',
                    '.au': 'audio/basic',
                    '.aiff': 'audio/aiff'
                }
                audio_mime = mime_map.get(ext.lower(), 'audio/wav')
                audio_uri = f"data:{audio_mime};base64,{audio_b64}"

                duration_sec = round(float(len(y)) / float(sr), 2)
                tempo_val = round(float(meta.get('tempo', 0.0)), 1)
                beats_val = int(meta.get('raw_beats', meta.get('beats', 0)))

                context = {
                    'genre': prediction_result['predicted_genre'].capitalize(),
                    'genre_raw': prediction_result['predicted_genre'],
                    'top_confidence': prediction_result['top_confidence'],
                    'top3': prediction_result['top3'],
                    'all_probabilities': prediction_result['all_probabilities'],
                    'model_type': prediction_result['model_type'],
                    'architecture': prediction_result.get('architecture', architecture),
                    'model_display_name': prediction_result['model_display_name'],
                    'waveform_b64': waveform_b64,
                    'spectrogram_b64': spectrogram_b64,
                    'audio_uri': audio_uri,
                    'audio_mime': audio_mime,
                    'filename': uploadfile.name,
                    'duration': duration_sec,
                    'sample_rate': sr,
                    'tempo': tempo_val,
                    'beats': beats_val,
                }
                return render(request, 'music/result.html', context)

            except Exception as e:
                messages.error(request, f'Audio analysis failed: {str(e)}')
                return redirect("predictor:index")

        else:
            messages.error(request, 'Invalid form submission. Please try again.')
            return redirect("predictor:index")

    # If accessed via GET, redirect to the index upload page
    return redirect("predictor:index")
