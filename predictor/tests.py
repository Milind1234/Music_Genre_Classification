import os
import io
import soundfile as sf
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

TEST_WAV = 'sample_audio/3.wav' if os.path.exists('sample_audio/3.wav') else '3.wav'


class PredictorViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_index_view_status_code(self):
        """Index page should load successfully."""
        response = self.client.get(reverse('predictor:index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'CHECK MUSIC GENRE')

    def test_upload_invalid_extension(self):
        """Non-audio extensions should be rejected with an error message."""
        fake_file = SimpleUploadedFile("sample.txt", b"plain text content", content_type="text/plain")
        response = self.client.post(reverse('predictor:result'), {'document': fake_file}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Unsupported file format")

    def test_upload_no_file(self):
        """Submitting an empty form should redirect back to index."""
        response = self.client.post(reverse('predictor:result'), {}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Please select or drag")

    def test_indian_music_model_classification(self):
        """Default Indian music model should return one of the 8 Indian genres."""
        with open(TEST_WAV, 'rb') as f:
            audio_bytes = f.read()

        uploaded = SimpleUploadedFile('3.wav', audio_bytes, content_type='audio/wav')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'indian'
        })

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'music/result.html')
        self.assertIn('genre', response.context)
        self.assertIn('top3', response.context)
        self.assertEqual(response.context['model_type'], 'indian')
        indian_genres = ['Bollywood', 'Carnatic', 'Ghazal', 'Semiclassical', 'Sufi', 'Bhajan', 'Folk', 'Hindustani', 'Punjabi']
        self.assertIn(response.context['genre'], indian_genres)
        self.assertEqual(len(response.context['top3']), 3)

    def test_mp3_audio_upload_support(self):
        """MP3 audio format should be accepted and classified."""
        mp3_sample = 'datasets/kaggle_indian/bollypop/bp01.mp3'
        if os.path.exists(mp3_sample):
            with open(mp3_sample, 'rb') as f:
                audio_bytes = f.read()
            uploaded = SimpleUploadedFile('bp01.mp3', audio_bytes, content_type='audio/mpeg')
            response = self.client.post(reverse('predictor:result'), {
                'document': uploaded,
                'model_type': 'indian'
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.context['genre'], 'Bollywood')
            self.assertIn('waveform_b64', response.context)
            self.assertIn('spectrogram_b64', response.context)

    def test_global_music_model_classification(self):
        """Global model should return 200 with GTZAN genres (Classical for 3.wav)."""
        with open(TEST_WAV, 'rb') as f:
            audio_bytes = f.read()

        uploaded = SimpleUploadedFile('3.wav', audio_bytes, content_type='audio/wav')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'global'
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['model_type'], 'global')
        self.assertEqual(response.context['genre'], 'Classical')
        self.assertIn('waveform_b64', response.context)
        self.assertIn('spectrogram_b64', response.context)

    def test_flac_audio_upload_support(self):
        """FLAC audio format should be accepted and classified."""
        with open(TEST_WAV, 'rb') as f:
            y, sr = sf.read(f)
        bio = io.BytesIO()
        sf.write(bio, y, sr, format='FLAC')
        uploaded = SimpleUploadedFile('sample.flac', bio.getvalue(), content_type='audio/flac')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'global'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['genre'], 'Classical')

    def test_cnn_deep_learning_classification(self):
        """Deep learning 2D CNN model should return 200 with Indian genres and audio playback."""
        with open(TEST_WAV, 'rb') as f:
            audio_bytes = f.read()

        uploaded = SimpleUploadedFile('sample.wav', audio_bytes, content_type='audio/wav')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'indian',
            'architecture': 'cnn'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['architecture'], 'cnn')
        self.assertIn('audio_uri', response.context)
        self.assertIn('waveform_b64', response.context)

    def test_ogg_audio_upload_support(self):
        """OGG audio format should be accepted and classified."""
        with open(TEST_WAV, 'rb') as f:
            y, sr = sf.read(f)
        bio = io.BytesIO()
        sf.write(bio, y, sr, format='OGG')
        uploaded = SimpleUploadedFile('sample.ogg', bio.getvalue(), content_type='audio/ogg')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'global'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['genre'], 'Classical')

    def test_cnn_deep_learning_classification(self):
        """Deep learning 2D CNN model should return 200 with Indian genres and audio playback."""
        with open(TEST_WAV, 'rb') as f:
            audio_bytes = f.read()

        uploaded = SimpleUploadedFile('sample.wav', audio_bytes, content_type='audio/wav')
        response = self.client.post(reverse('predictor:result'), {
            'document': uploaded,
            'model_type': 'indian',
            'architecture': 'cnn'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['architecture'], 'cnn')
        self.assertIn('audio_uri', response.context)
        self.assertIn('waveform_b64', response.context)
