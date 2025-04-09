import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import sounddevice as sd
import os
import tempfile
from io import BytesIO

# Constants
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_SEGMENTS = 10

# Load your pre-trained model (replace with your actual model path)
# For this example, we'll create a dummy model if none exists
def load_or_create_model():
    model_path = "mfcc_lstm_model.h5"
    if not os.path.exists(model_path):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
        
        # Create a dummy model for demonstration
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(N_SEGMENTS, N_MFCC)),
            Dropout(0.3),
            SimpleRNN(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(model_path)
        st.warning("Using a dummy model. Please train a real model for accurate results.")
    return load_model(model_path)

model = load_or_create_model()

def process_audio_file(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Pad or trim audio to desired duration
        if len(y) > SAMPLE_RATE * DURATION:
            y = y[:SAMPLE_RATE * DURATION]
        else:
            padding = SAMPLE_RATE * DURATION - len(y)
            y = np.pad(y, (0, padding))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        
        # Split into segments
        segments = []
        num_samples_per_segment = int(len(mfcc) / N_SEGMENTS)
        
        for s in range(N_SEGMENTS):
            start = num_samples_per_segment * s
            finish = start + num_samples_per_segment
            segments.append(mfcc[start:finish])
        
        return np.array(segments)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def record_audio(duration=4, sample_rate=22050):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return recording.flatten()

def save_audio_to_tempfile(audio, sample_rate=22050):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    librosa.output.write_wav(temp_file.name, audio, sample_rate)
    return temp_file.name

def plot_waveform_and_mfcc(audio, sample_rate):
    fig, ax = plt.subplots(2, figsize=(10, 7))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sample_rate, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    
    # Plot MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.f")
    ax[1].set_title('MFCC Coefficients')
    
    st.pyplot(fig)

def predict_audio(audio_path):
    features = process_audio_file(audio_path)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        prediction = model.predict(features)
        return prediction[0][0]  # Return probability of being fake
    return None

# Streamlit UI
st.title("Real/Fake Audio Detection")
st.write("This app uses MFCC features with LSTM/RNN to detect whether audio is real or fake.")

option = st.radio("Select input method:", ("Upload Audio File", "Record Audio"))

audio_path = None

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    if uploaded_file is not None:
        # Save to temp file
        if uploaded_file.name.endswith('.mp3'):
            audio = AudioSegment.from_mp3(uploaded_file)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio.export(temp_file.name, format="wav")
            audio_path = temp_file.name
        else:
            audio_path = save_audio_to_tempfile(librosa.load(uploaded_file, sr=SAMPLE_RATE)[0])
        
        st.audio(uploaded_file)
        
        # Process and display
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        plot_waveform_and_mfcc(y, sr)
        
        if st.button("Analyze Audio"):
            with st.spinner("Processing..."):
                prediction = predict_audio(audio_path)
                if prediction is not None:
                    st.subheader("Results")
                    st.write(f"Probability of being fake: {prediction:.4f}")
                    if prediction > 0.5:
                        st.error("Prediction: FAKE audio (probability > 0.5)")
                    else:
                        st.success("Prediction: REAL audio (probability ≤ 0.5)")

else:  # Record Audio
    if st.button("Start Recording"):
        audio = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
        audio_path = save_audio_to_tempfile(audio)
        
        st.audio(audio_path)
        plot_waveform_and_mfcc(audio, SAMPLE_RATE)
        
        if st.button("Analyze Recording"):
            with st.spinner("Processing..."):
                prediction = predict_audio(audio_path)
                if prediction is not None:
                    st.subheader("Results")
                    st.write(f"Probability of being fake: {prediction:.4f}")
                    if prediction > 0.5:
                        st.error("Prediction: FAKE audio (probability > 0.5)")
                    else:
                        st.success("Prediction: REAL audio (probability ≤ 0.5)")

# Clean up temp files
if audio_path and os.path.exists(audio_path):
    os.unlink(audio_path)
