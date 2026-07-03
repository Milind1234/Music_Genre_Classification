def getmetadata(filename):
    import librosa
    import numpy as np

    # If Django uploaded file object is passed, make sure reading starts from beginning
    if hasattr(filename, "seek"):
        filename.seek(0)

    y, sr = librosa.load(filename, sr=22050)

    # tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

    # beats count
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = float(len(beat_frames))

    # other audio features
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
        'chroma_stft': float(np.mean(chroma_stft)),
        'rmse': float(np.mean(rmse)),
        'spectral_centroid': float(np.mean(spectral_centroid)),
        'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
        'rolloff': float(np.mean(rolloff)),
        'zero_crossing_rate': float(np.mean(zero_crossing_rate))
    }

    for i in range(1, 21):
        metadata_dict[f'mfcc{i}'] = float(np.mean(mfcc[i-1]))

    return metadata_dict