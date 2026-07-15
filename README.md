# BirdyCall

Offline bird sound recognition. Records a short clip of a bird call, runs it through a locally-trained CNN-LSTM neural network, and predicts which bird species it is, no internet connection required at prediction time.
Check out the demo video of the app here : https://drive.google.com/drive/folders/1jZiXwdnv7oYauz6fwS8dbtfxEa8TM1sg?usp=sharing
## 1. Overview

The pipeline has four stages:

1. **Dataset**: real bird recordings sorted into one folder per species
2. **Preprocessing**: WAV audio converted into mel spectrograms (image-like representations of sound)
3. **Training**: a CNN-LSTM hybrid model learns to classify spectrograms by species
4. **Conversion**: the trained model is compressed into TensorFlow Lite format for fast, lightweight inference

A separately recorded microphone clip is preprocessed the same way and run through the trained model to get a prediction.

## 2. Setup Instructions

1. Install [Python](https://www.python.org/downloads/) 
2. Clone or download this repository.
3. From inside the project folder, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Populate `dataset/<Bird_Name>/` with real WAV recordings per species required.

## 3. Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Builds, trains, and runs the CNN-LSTM model |
| `librosa` | Loads audio, trims silence, generates mel spectrograms |
| `numpy` | Array/numeric operations throughout the pipeline |
| `scipy` | Supporting signal-processing utilities |
| `soundfile` | Reads/writes WAV files |
| `scikit-learn` | Train/validation splitting |
| `sounddevice` | Records audio from the laptop's microphone |

All are pinned in `requirements.txt` and installed with `pip install -r requirements.txt`. No paid APIs or cloud services are required at any stage.

## 4. Run Commands, Sample Inputs, and Expected Outputs

Run these in order from the project root.

### Step 1 Populate the dataset

Download real recordings (e.g. from Xeno-canto):
- drop short (3–5s) clips directly into `dataset/<Bird_Name>/`, or
- run long/raw recordings through the splitter first:
  ```bash
  python split_recordings.py downloads/raw_file.wav dataset/House_Sparrow
  ```
  **Sample input:** a 60-second raw WAV recording of a house sparrow.
  **Expected output:** several short WAV clips (e.g. `raw_file_clip000.wav`, `raw_file_clip001.wav`, ...) saved into `dataset/House_Sparrow/`, one per detected call.
  
**Expected output:** a per-species clip count table, flagging any empty or heavily imbalanced folders.

### Step 2 Train the model

```bash
python train.py
```
This internally handles preprocessing, there is no separate preprocessing command to run.

**Sample input:** whatever WAV clips are currently sitting in `dataset/<Bird_Name>/`.
**Expected output:**
```
Step 1/4: Building dataset from dataset/ folder...
Step 2/4: Splitting into train/validation sets...
Step 3/4: Building and training the CNN-LSTM...
Step 4/4: Saving model and labels...

Done. Final validation accuracy: XX.X%
Model saved to: bird_model.keras
Labels saved to: labels.txt
```

### Step 3 Convert to TensorFlow Lite

```bash
python convert_to_tflite.py
```
**Expected output:** `bird_model.tflite`, roughly 10–12x smaller than `bird_model.keras`, with a printed size comparison.

### Step 4 Predict from a saved WAV file

```bash
python predict.py sample_audio/some_bird.wav
```
**Sample input:** a single WAV clip, or a folder of them.
**Expected output:**
```
Audio: sample_audio/some_bird.wav
Prediction: House_Sparrow
Confidence: 94.2%
```

To test the converted `.tflite` model specifically (verifying it agrees with the original):
```bash
python predict_tflite.py sample_audio/some_bird.wav
```

### Step 5 Predict live from the laptop microphone

This is the actual end-user flow: a short clip is recorded live from the laptop's built-in (or connected) microphone, run through the same preprocessing and trained model used above, and a prediction is printed, no pre-saved WAV file needed for this step.

**Sample input:** ambient sound captured live for a fixed clip duration by the laptop's microphone.
**Expected output:** the predicted species name and a confidence percentage, printed the moment the clip finishes recording and processing.

## Local AI Verification

| Stage | Runs on-device? | Needs internet? |
|---|---|---|
| Recording audio from the mic | Yes, entirely local | No |
| Preprocessing (spectrogram generation) | Yes, entirely local | No |
| Model inference (prediction) | Yes, entirely local | No |
| Model training | Yes, entirely local | No (dataset must already be downloaded) |
| Sourcing the training dataset | N/A | Yes, one-time, to download recordings from Xeno-canto/Macaulay/BirdCLEF |

Once the dataset is downloaded and the model is trained, **the entire prediction pipeline: recording, preprocessing, and inference, it runs fully on-device with no network calls.** No audio, predictions, or other data are ever sent off the device during normal operation. Internet access is only ever used once, manually, by the developer, to fetch training recordings from public archives, it plays no role in the running application itself.

## Privacy and Safety

**Data handling:** Microphone audio is processed transiently in memory (and briefly as a temporary WAV file during preprocessing) purely to generate a prediction. No audio recordings, spectrograms, or predictions are logged, uploaded, or persisted beyond what the user explicitly saves themselves.

**Permissions:** The application requires microphone access on whatever device it runs on. No other device permissions (location, camera, contacts, storage beyond the project folder, etc.) are used.

**Storage:** Only the training dataset (`dataset/`), the trained model files (`bird_model.keras`, `bird_model.tflite`), and `labels.txt` are stored persistently, all locally on disk. No cloud storage or external database is used.

**Limitations:**
- The model only recognizes the species it was trained on; anything else (other birds, human speech, background noise) will still be forced into one of those categories with some confidence score, which can be misleading.
- Accuracy depends heavily on training data quality and diversity, a class trained on few or narrow-source recordings will generalize poorly to new environments.
- Real-world background noise (traffic, wind, overlapping bird calls) was not systematically modeled during training and may reduce accuracy versus clean recordings.
- Not intended for scientific or conservation-grade species identification, this is an educational/prototype system.

## Attribution

**Pretrained models:** None. The CNN-LSTM classifier in `train.py` is trained entirely from scratch on the project's own dataset, no pretrained weights or transfer learning are used.

**Datasets (recordings sourced from, not redistributed in this repo):**
- [Xeno-canto](https://xeno-canto.org/): primary source of bird call recordings
- [BirdCLEF](https://www.kaggle.com/competitions/birdclef-2024) (Kaggle): supplementary/reference dataset

**Libraries and frameworks:**
- [TensorFlow / Keras](https://www.tensorflow.org/): model architecture, training, TFLite conversion
- [librosa](https://librosa.org/): audio loading, silence trimming, mel spectrogram generation
- [scikit-learn](https://scikit-learn.org/): train/validation splitting
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/): numerical and signal processing
- [SoundFile](https://python-soundfile.readthedocs.io/): WAV file I/O
- [sounddevice](https://python-sounddevice.readthedocs.io/): live microphone recording

**APIs:** None. The system does not call any external API at prediction time or otherwise.
