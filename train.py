import os
import numpy as np
from src.speech_to_text import audio_to_text
from src.grammar_features import extract_grammar_features
from src.model import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/audio"

X, y = [], []

for file in os.listdir(DATA_PATH):
    if file.endswith(".wav"):
        text = audio_to_text(os.path.join(DATA_PATH, file))
        features = extract_grammar_features(text)
        X.append(features)
        y.append(max(0, 10 - features[0]))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
