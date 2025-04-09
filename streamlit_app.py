import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # You can also extract a Mel-spectrogram or any other feature
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    return mfcc.mean(axis=1)  # Average MFCC values across time frames
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Define a simple CNN model for audio classification
class AudioClassifier(nn.Module):
    def init(self, input_size):
        super(AudioClassifier, self).init()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_size // 2 - 2), 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: real or fake
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example to load and preprocess the data
def load_data(audio_files, labels):
    features = []
    for file in audio_files:
        feature = extract_features(file)
        features.append(feature)
    return np.array(features), np.array(labels)

# Example of training
def train_model():
    # Example dataset (You should replace with your real/fake audio dataset)
    audio_files = ["audio1.wav", "audio2.wav"]  # Replace with actual paths
    labels = [0, 1]  # 0 for real, 1 for fake (just an example)

    X, y = load_data(audio_files, labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Convert to torch tensors
    X_train = torch.tensor(X_train).float().unsqueeze(1)  # Add channel dimension
    y_train = torch.tensor(y_train).long()
    X_val = torch.tensor(X_val).float().unsqueeze(1)
    y_val = torch.tensor(y_val).long()

    # Initialize the model
    model = AudioClassifier(input_size=X_train.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):  # 20 epochs for example
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), "audio_classifier.pth")

    return model

import streamlit as st
import torch
from scipy.io.wavfile import write

# Load the trained model
model = AudioClassifier(input_size=13)  # Use the right input size
model.load_state_dict(torch.load("audio_classifier.pth"))
model.eval()

def predict_audio(file):
    # Extract features from the uploaded audio file
    features = extract_features(file)
    features = torch.tensor(features).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Make prediction
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, dim=1)
        return "Fake" if prediction == 1 else "Real"

def main():
    st.title("Real vs Fake Audio Classifier")
    st.markdown("Upload an audio file to determine if it is real or fake.")
    
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        # Save the uploaded file temporarily to predict on it
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        result = predict_audio("temp_audio.wav")
        st.write(f"Prediction: {result}")

if __name__ == "__main__":
    main()
