import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tempfile
import os
import pickle
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
from io import BytesIO

# Constants
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_SEGMENTS = 10

# Configuration
st.set_page_config(page_title="Audio Authenticity Detector", layout="wide")

@st.cache_resource
def load_model():
    """Load or create the LSTM model"""
    model_path = "audio_model.h5"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path):
        # Create a dummy model if none exists
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(N_SEGMENTS, N_MFCC)),
            Dropout(0.3),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(model_path)
        
        # Create dummy scaler
        scaler = StandardScaler()
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        st.warning("Using a dummy model. For accurate results, train with real data.")
    
    model = Sequential()
    model.load_weights(model_path)
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = StandardScaler()
    
    return model, scaler

def extract_mfcc(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Pad or trim audio
        if len(y) > SAMPLE_RATE * DURATION:
            y = y[:SAMPLE_RATE * DURATION]
        else:
            padding = SAMPLE_RATE * DURATION - len(y)
            y = np.pad(y, (0, padding))
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, 
            n_fft=n_fft, hop_length=hop_length
        )
        return mfcc.T
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def prepare_segments(mfcc_features, n_segments=10):
    """Prepare segments for model prediction"""
    segments = []
    num_samples_per_segment = int(len(mfcc_features) / n_segments
    
    for s in range(n_segments):
        start = int(num_samples_per_segment * s)
        finish = int(start + num_samples_per_segment)
        segments.append(mfcc_features[start:finish])
    
    return np.array(segments)

def plot_audio_features(y, sr):
    """Plot waveform and MFCC features"""
    fig, ax = plt.subplots(2, figsize=(10, 7))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.f")
    ax[1].set_title('MFCC Coefficients')
    
    st.pyplot(fig)

def main():
    st.title("🎙️ Audio Authenticity Detector")
    st.write("""
    This system detects whether an audio recording is genuine or synthetic/synthetic
    using MFCC features and LSTM neural networks.
    """)
    
    # Load model and scaler
    model, scaler = load_model()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.5, max_value=0.99, value=0.75, step=0.01
        )
        st.markdown("---")
        st.info("""
        **Instructions:**
        1. Record or upload audio (4 seconds recommended)
        2. Click 'Analyze Audio'
        3. View results and visualizations
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Audio")
        tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
        
        with tab1:
            audio_data = st_audiorec()
            
        with tab2:
            uploaded_file = st.file_uploader(
                "Choose an audio file", 
                type=["wav", "mp3", "ogg", "flac"]
            )
            if uploaded_file:
                if uploaded_file.name.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(uploaded_file)
                    buffer = BytesIO()
                    audio.export(buffer, format="wav")
                    audio_data = buffer.getvalue()
                else:
                    audio_data = uploaded_file.getvalue()
    
    with col2:
        st.header("Analysis Results")
        
        if audio_data is not None:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            # Display audio player
            st.audio(audio_data, format="audio/wav")
            
            if st.button("Analyze Audio", use_container_width=True):
                with st.spinner("Processing audio..."):
                    try:
                        # Load and process audio
                        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
                        
                        # Plot features
                        plot_audio_features(y, sr)
                        
                        # Extract features
                        mfcc_features = extract_mfcc(tmp_path)
                        
                        if mfcc_features is not None:
                            # Prepare segments
                            segments = prepare_segments(mfcc_features)
                            
                            # Scale features
                            segments_scaled = scaler.transform(
                                segments.reshape(-1, N_MFCC)
                            ).reshape(segments.shape)
                            
                            # Make prediction
                            prediction = model.predict(
                                np.expand_dims(segments_scaled, axis=0)
                            )[0][0]
                            
                            # Display results
                            st.subheader("Detection Results")
                            confidence = prediction if prediction > 0.5 else 1 - prediction
                            
                            if prediction > confidence_threshold:
                                st.error(f"⚠️ Synthetic Audio Detected (confidence: {confidence:.2%})")
                                st.progress(float(confidence))
                                st.markdown("""
                                **Characteristics detected:**
                                - Potential signs of voice synthesis
                                - Artificial speech patterns
                                """)
                            elif prediction > 0.5:
                                st.warning(f"🤔 Possibly Synthetic (confidence: {confidence:.2%})")
                                st.progress(float(confidence))
                            else:
                                st.success(f"✅ Genuine Audio (confidence: {confidence:.2%})")
                                st.progress(float(confidence))
                                st.markdown("""
                                **Characteristics detected:**
                                - Natural speech patterns
                                - Consistent with human voice
                                """)
                            
                            # Show raw prediction value
                            with st.expander("Technical Details"):
                                st.write(f"Raw prediction value: {prediction:.4f}")
                                st.write(f"MFCC shape: {mfcc_features.shape}")
                                st.write(f"Segments shape: {segments.shape}")
                                
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                    finally:
                        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
