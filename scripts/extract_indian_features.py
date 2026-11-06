"""
Extract Indian Audio Features from Multiple Datasets
Extracts 28 numerical acoustic features across Indian music genres:
- Bollywood (commercial pop/filmi)
- Carnatic (South Indian Classical)
- Ghazal (Poetic/traditional)
- Semiclassical (Semi-classical art music)
- Sufi (Devotional Qawwali/spiritual)
- Bhajan (Hindu devotional hymns)
- Folk (Regional Indian Folk: Bhojpuri, Garhwali)
- Desi Hip-Hop (Bollywood rap & urban beats)
"""

import os
import glob
import sys
import time
from multiprocessing import Pool, cpu_count
import pandas as pd

# Automatically add project root to python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure matplotlib writes to /tmp to avoid permissions issues
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

# Import feature extraction logic
from predictor.Metadata import getmetadata

# Genre mapping to standardize classes
GENRE_MAP = {
    'bollypop': 'bollywood',
    'bollywood_romantic': 'bollywood',
    'bollywood_rap': 'desi_hiphop',
    'semiclassical': 'semiclassical',
    'carnatic': 'carnatic',
    'ghazal': 'ghazal',
    'sufi': 'sufi',
    'bhajan': 'bhajan',
    'bhojpuri': 'folk',
    'garhwali': 'folk',
}


def _process_single_track(item):
    """Worker function for multiprocessing."""
    file_path, genre_name = item
    try:
        meta = getmetadata(file_path)
        meta['filename'] = os.path.basename(file_path)
        meta['class_name'] = genre_name
        meta['label'] = genre_name
        return meta
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(file_path)}: {e}")
        return None


def extract_features_parallel(input_dirs=None, output_csv=None, workers=None, max_per_genre=100):
    if input_dirs is None:
        input_dirs = [
            os.path.join(PROJECT_ROOT, "datasets/kaggle_indian"),
            os.path.join(PROJECT_ROOT, "datasets/gacmis/Dataset/Songs")
        ]
    elif isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    if output_csv is None:
        output_csv = os.path.join(PROJECT_ROOT, "datasets/indian_music_data.csv")

    if workers is None:
        workers = min(8, cpu_count())

    supported_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.opus')
    genre_file_map = {}

    print("Scanning audio tracks across datasets:")
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"  [SKIP] Directory not found: {directory}")
            continue

        print(f"  Scanning: {directory}")
        for genre_folder in sorted(os.listdir(directory)):
            genre_path = os.path.join(directory, genre_folder)
            if not os.path.isdir(genre_path):
                continue

            genre_raw = genre_folder.lower().strip()
            standardized_genre = GENRE_MAP.get(genre_raw, genre_raw)

            audio_files = [
                f for f in glob.glob(os.path.join(genre_path, "*"))
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]

            if standardized_genre not in genre_file_map:
                genre_file_map[standardized_genre] = []
            genre_file_map[standardized_genre].extend(audio_files)

    # Queue tasks with per-genre balancing
    tasks = []
    print("\nQueued tracks per genre:")
    for genre, files in sorted(genre_file_map.items()):
        selected = files[:max_per_genre]
        for f in selected:
            tasks.append((f, genre))
        print(f"  - {genre:15}: {len(selected):3d} tracks (from {len(files)} total found)")

    print(f"\nTotal tracks queued for extraction: {len(tasks)}")
    print(f"Starting parallel extraction using {workers} CPU workers...")

    start_time = time.time()
    rows = []

    with Pool(processes=workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(_process_single_track, tasks), 1):
            if result is not None:
                rows.append(result)
            if idx % 25 == 0 or idx == len(tasks):
                elapsed = time.time() - start_time
                print(f"  [{idx:3d}/{len(tasks)}] extracted ({elapsed:.1f}s elapsed)...")

    elapsed_total = time.time() - start_time
    print(f"\nCompleted extraction of {len(rows)} tracks in {elapsed_total:.1f}s ({len(rows)/max(elapsed_total, 1):.1f} tracks/sec)!")

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved dataset to {output_csv}")

        print("\nFinal Genre Distribution:")
        print(df['label'].value_counts())
        return df
    else:
        print("\n[ERROR] No valid features extracted.")
        return None


if __name__ == '__main__':
    extract_features_parallel()
