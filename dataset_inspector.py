import os
import wave
import glob
from collections import defaultdict

data_dir = 'c:/Users/moham/OneDrive/Desktop/CodeAlpha/CodeAlpha_Emotion_Recognition/data/RAVDESS'

emotions_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("=== RAVDESS DATASET REPORT ===")

# 1. Number of actor folders
actor_folders = [f for f in os.listdir(data_dir) if f.startswith('Actor_') and os.path.isdir(os.path.join(data_dir, f))]
print(f"1. Number of actor folders: {len(actor_folders)}")

# Analyze files
total_wavs = 0
files_per_actor = {}
emotion_counts = defaultdict(int)
sample_filenames = []
all_filenames = set()
duplicates = []
corrupt_files = []

channels_set = set()
sample_rate_set = set()
durations = []

for actor in sorted(actor_folders):
    actor_path = os.path.join(data_dir, actor)
    wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
    total_wavs += len(wav_files)
    files_per_actor[actor] = len(wav_files)
    
    for f in wav_files:
        if f in all_filenames:
            duplicates.append(f)
        all_filenames.add(f)
        
        # 4. Extract emotion label
        parts = f.replace('.wav', '').split('-')
        if len(parts) == 7:
            emotion_code = parts[2]
            emotion = emotions_map.get(emotion_code, 'unknown')
            emotion_counts[emotion] += 1
            
            if len(sample_filenames) < 10:
                sample_filenames.append((f, emotion))
                
        # 8 & 9. Check for corruption and get audio properties
        file_path = os.path.join(actor_path, f)
        try:
            with wave.open(file_path, 'rb') as w:
                channels_set.add(w.getnchannels())
                sample_rate_set.add(w.getframerate())
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
                durations.append(duration)
        except Exception as e:
            corrupt_files.append(f)

# 10. Check if data/audio_speech_actors_01-24 duplicates Actor_01 to Actor_24
dup_dir = os.path.join(data_dir, 'audio_speech_actors_01-24')
has_dup_dir = os.path.isdir(dup_dir)
if has_dup_dir:
    dup_files = []
    for root, dirs, files in os.walk(dup_dir):
        for f in files:
            if f.endswith('.wav'):
                dup_files.append(f)
    dup_is_duplicate = all(f in all_filenames for f in dup_files) and len(dup_files) > 0
else:
    dup_is_duplicate = False

print(f"2. Total number of WAV files: {total_wavs}")
print(f"3. Number of files per actor: {files_per_actor}")
print(f"4. Emotion label convention: Found by splitting filename by '-' and taking the 3rd element.")
print(f"5. Unique emotion labels and counts: {dict(emotion_counts)}")
print("6. Sample filenames and decoded labels:")
for sf, em in sample_filenames:
    print(f"   {sf} -> {em}")
print(f"7. Duplicate filenames found: {len(duplicates)}")
print(f"8. Corrupted WAV files found: {len(corrupt_files)}")
if durations:
    avg_dur = sum(durations) / len(durations)
else:
    avg_dur = 0
print(f"9. Audio format: Channels={channels_set}, Sample Rate={sample_rate_set}Hz, Avg Duration={avg_dur:.2f}s")
print(f"10. `audio_speech_actors_01-24` duplicates the Actor folders? {dup_is_duplicate} (Files in it: {len(dup_files) if has_dup_dir else 0})")
