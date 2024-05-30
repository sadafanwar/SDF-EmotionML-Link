# pip install torch torchvision opencv-python numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

# Define the CNN3D class
class CNN3D(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.dropout1 = nn.Dropout3d(0.25)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.dropout2 = nn.Dropout3d(0.25)
        self.fc1 = nn.Linear(401408, 512)  # Make sure the input size is correct after flattening
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the classes
class_names_8 = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define the mapping to character types
emotion_class_to_character = {
    0: ["Poe"],
    1: ["Trickster", "MHBFY Jenny", "Achiever"],
    2: ["Chatroom Bob", "Flirt", "Troll", "Wizard"],
    3: ["Ripper", "Elder"],
    4: ["Haters", "E-Venger", "Godwin", "Big Man"],
    5: ["Lurker"],
    6: ["Iconoclast", "Snert"],
    7: ["Sorcerer"]
}

character_details = {
    "Haters": (4, ":-}}", "🧔‍♂️"),
    "E-Venger": (4, ":@" , "🤯"),
    "Iconoclast": (6, "^o)", "🙄"),
    "Snert": (6, "8o)", "😎"),
    "Godwin": (4, ":-{", "🤔"),
    "Trickster": (1, ":-∞", "👦"),
    "Big Man": (4, "-)", "🤔"),
    "Chatroom Bob": (2, ";-)", "😉"),
    "Poe": (0, "8-{", "🧐"),
    "Ripper": (3, ":-(", "😔"),
    "MHBFY Jenny": (1, ":-o", "😮"),
    "Flirt": (2, ":-)", "😊"),
    "Sorcerer": (7, ":ℵ", "😳"),
    "Lurker": (5, ":-#", "🤐"),
    "Troll": (2, ":-D", "😁"),
    "Elder": (3, ":-", "😔"),
    "Achiever": (1, ":-0", "👴"),
    "Wizard": (2, "8-)", "😎")
}

# Load the pre-trained model
model_path = 'emotion_classification_simple_3dcnn.pth'
model = CNN3D(num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to capture and preprocess video frames
def capture_frames(cap, num_frames=6):
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))  # Resize frame to match the input size
        frame_tensor = transform(frame_resized)
        frames.append(frame_tensor)
    if len(frames) < num_frames:
        frames = (frames * ((num_frames // len(frames)) + 1))[:num_frames]
    frames = torch.stack(frames, dim=0)
    frames = frames.permute(1, 0, 2, 3)
    return frames.unsqueeze(0)  # Add batch dimension

# Function to map detected class to the character details
def get_character_details(detected_class):
    character_types = emotion_class_to_character[detected_class]
    return [character_details[char] for char in character_types]

# Main loop to capture and predict emotion
try:
    while True:
        frames = capture_frames(cap)
        with torch.no_grad():
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            detected_emotion = preds[0].item()
            character_details_list = get_character_details(detected_emotion)

        # Display the resulting frame
        ret, frame = cap.read()
        if ret:
            y_position = 30
            for details in character_details_list:
                emotion_class, emoticon, avatar = details
                text = f'{class_names_8[detected_emotion]} ({emotion_class}): {emoticon} {avatar}'
                cv2.putText(frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y_position += 30
            cv2.imshow('Webcam Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
