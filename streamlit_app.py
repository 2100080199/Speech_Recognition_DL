import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pydub import AudioSegment
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

# Load or create model (same as before)
def load_or_create_model():
    model_path = "mfcc_lstm_model.h5"
    if not os.path.exists(model_path):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
        
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

# Rest of your processing functions remain the same...

# Modified Streamlit UI without sounddevice
st.title("Real/Fake Audio Detection")
st.write("This app uses MFCC features with LSTM/RNN to detect whether audio is real or fake.")

option = st.radio("Select input method:", ("Upload Audio File", "Use Streamlit Recorder"))

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

else:  # Use Streamlit Recorder
    audio_bytes = st.audio_recorder("Click to record", sample_rate=SAMPLE_RATE)
    if audio_bytes:
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes)
        temp_file.close()
        audio_path = temp_file.name
        
        st.audio(audio_bytes)
        
        # Process and display
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        plot_waveform_and_mfcc(y, sr)
        
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

# Clean up
if audio_path and os.path.exists(audio_path):
    os.unlink(audio_path)
