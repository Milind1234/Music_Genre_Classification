import os
import io
import tempfile
import base64
import numpy as np
import librosa
import librosa.display
try:
    import librosa.feature.rhythm
except ImportError:
    pass

# Configure matplotlib headless backend
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_with_av(source, target_sr=22050):
    """Fallback decoder using PyAV (supports .opus, webm, and complex codecs)."""
    try:
        import av  
        container = av.open(source)
        if not container.streams.audio:
            container.close()
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
        frames = []
        for frame in container.decode(stream):
            for resampled_frame in resampler.resample(frame):
                frames.append(resampled_frame.to_ndarray())
        container.close()
        if frames:
            return np.concatenate(frames, axis=1).squeeze(), target_sr
    except Exception:
        pass
    return None


def load_audio_flexible(filename, sr=22050):
    """
    Loads audio from either a file path or an in-memory/uploaded file object.
    Supports .wav, .mp3, .ogg, .flac, .m4a, .aac, .au, .aiff, .opus.
    Falls back to PyAV and temporary files if an in-memory stream cannot be decoded directly.
    """
    if hasattr(filename, "seek"):
        filename.seek(0)

    # Check if filename is an opus file or container
    fname = getattr(filename, 'name', '') if hasattr(filename, 'name') else str(filename)
    if fname.lower().endswith('.opus'):
        av_res = _load_with_av(filename, target_sr=sr)
        if av_res is not None:
            return av_res

    # Try direct load first
    try:
        y, loaded_sr = librosa.load(filename, sr=sr)
        return y, loaded_sr
    except Exception:
        # Fallback to PyAV before writing temp files
        if hasattr(filename, "seek"):
            filename.seek(0)
        av_res = _load_with_av(filename, target_sr=sr)
        if av_res is not None:
            return av_res

        # Fallback to saving to a named temporary file with the proper extension
        ext = ".wav"
        if hasattr(filename, "name") and filename.name:
            _, file_ext = os.path.splitext(filename.name)
            if file_ext:
                ext = file_ext.lower()

        if hasattr(filename, "seek"):
            filename.seek(0)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            if hasattr(filename, "read"):
                tmp.write(filename.read())
            elif isinstance(filename, (bytes, bytearray)):
                tmp.write(filename)
            tmp_path = tmp.name

        try:
            y, loaded_sr = librosa.load(tmp_path, sr=sr)
            return y, loaded_sr
        except Exception:
            av_res = _load_with_av(tmp_path, target_sr=sr)
            if av_res is not None:
                return av_res
            raise
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def getmetadata(filename, return_audio=False):
    """
    Extracts the 28 numerical features required for music genre classification.
    If return_audio is True, returns (metadata_dict, y, sr).
    """
    y, sr = load_audio_flexible(filename, sr=22050)

    # Tempo calculation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    except AttributeError:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

    # Audio duration in seconds
    duration = float(len(y)) / float(sr)

    # Beats count
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    raw_beats = float(len(beat_frames))
    # Standardize beats to a 30-second reference window (GTZAN standard).
    # This ensures songs of any duration (30s to 6+ minutes) have comparable, normalized beat features.
    if duration > 0:
        beats = raw_beats * (30.0 / duration)
    else:
        beats = raw_beats

    # Spectral and energy features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    metadata_dict = {
        'tempo': float(tempo),
        'beats': float(beats),
        'raw_beats': float(raw_beats),
        'duration': float(duration),
        'chroma_stft': float(np.mean(chroma_stft)),
        'rmse': float(np.mean(rmse)),
        'spectral_centroid': float(np.mean(spectral_centroid)),
        'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
        'rolloff': float(np.mean(rolloff)),
        'zero_crossing_rate': float(np.mean(zero_crossing_rate))
    }

    for i in range(1, 21):
        metadata_dict[f'mfcc{i}'] = float(np.mean(mfcc[i-1]))

    if return_audio:
        return metadata_dict, y, sr
    return metadata_dict


def generate_visualizations(y, sr):
    """
    Generates high-contrast styled waveform and mel-spectrogram base64 images.
    Returns (waveform_b64, spectrogram_b64).
    """
    # 1. Waveform Plot
    fig1, ax1 = plt.subplots(figsize=(9, 2.8), facecolor='#181c2e')
    ax1.set_facecolor('#181c2e')
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#818cf8', alpha=0.9)
    ax1.set_title('Audio Waveform', color='#e0e7ff', fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax1.set_ylabel('Amplitude', color='#94a3b8', fontsize=10)
    ax1.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')
    plt.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=110, facecolor=fig1.get_facecolor(), bbox_inches='tight')
    plt.close(fig1)
    waveform_b64 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    # 2. Mel-Spectrogram Plot
    fig2, ax2 = plt.subplots(figsize=(9, 3.2), facecolor='#181c2e')
    ax2.set_facecolor('#181c2e')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax2, cmap='magma')
    ax2.set_title('Mel-Spectrogram (Frequency vs Time)', color='#e0e7ff', fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel('Time (s)', color='#94a3b8', fontsize=10)
    ax2.set_ylabel('Hz', color='#94a3b8', fontsize=10)
    ax2.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    cbar = fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8', fontsize=9)
    cbar.outline.set_edgecolor('#334155')
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=110, facecolor=fig2.get_facecolor(), bbox_inches='tight')
    plt.close(fig2)
    spectrogram_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return waveform_b64, spectrogram_b64
