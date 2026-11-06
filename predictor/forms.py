from django import forms


class DocumentForm(forms.Form):
    def is_valid(*args, **kwargs):
        return True

    file = forms.FileField(
        required=False,
        help_text='Supported formats: .wav, .mp3, .ogg, .flac, .m4a, .aac, .au, .aiff'
    )
