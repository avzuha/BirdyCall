import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAMPLE_AUDIO_DIR = os.path.join(BASE_DIR, "sample_audio")
MODEL_PATH = os.path.join(BASE_DIR, "bird_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

SAMPLE_RATE = 22050
CLIP_DURATION = 4.0
N_MELS = 128              # number of mel bands (spectrogram "height")
N_FFT = 1024
HOP_LENGTH = 512
# Fixed number of time frames the spectrogram is resized to (the "width").
# With sr=22050, duration=4s, hop_length=512 -> ~173 frames
FIXED_FRAMES = 173
# this list must exactly match your dataset/ subfolder names
BIRD_CLASSES = [
    "House_Sparrow",
    "Rock_Pigeon",
    "House_Crow",
    "Common_Myna",
    "Rose_ringed_Parakeet",
    "Asian_Koel",
    "Red_vented_Bulbul",
    "Indian_Peafowl",
    "White_throated_Kingfisher",
    "Coppersmith_Barbet",
]
