# Import required libraries
import cv2  # OpenCV for image and video processing
import numpy as np  # For numerical computations
from keras.models import model_from_json  # To load the saved model structure and weights

# Emotion labels for the corresponding indices
emotions = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprised"
}

# Load the model architecture from a JSON file
json_file = open('machine_learning_model.json', 'r')  # Open the JSON file
loaded_model_json = json_file.read()  # Read the file content
json_file.close()  # Close the file
machine_learning_model = model_from_json(loaded_model_json)  # Load model architecture

# Load the pre-trained weights into the model
machine_learning_model.load_weights("machine_learning_model.weights.h5")
print("Loaded model from disk")

# Start capturing video from the webcam (0 is the default webcam index)
cap = cv2.VideoCapture(0)

# Uncomment this section to use a pre-recorded video instead of webcam feed
# Provide the video file path here:
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Read a single frame from the video capture
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))  # Resize the frame for consistency

    # Exit the loop if the frame is not captured correctly
    if not ret:
        break

    # Load Haar Cascade for face detection
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    num_faces = face_detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.3,  # How much the image size is reduced at each image scale
        minNeighbors=5  # Minimum neighbors each rectangle should have to retain it
    )

    # Process each detected face
    for (x, y, w, h) in num_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Crop and preprocess the region of interest (ROI)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]  # Extract the face area
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
        )  # Resize and add batch dimension

        # Predict the emotion for the cropped face
        emotion_prediction = machine_learning_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))  # Get the index of the highest probability

        # Add emotion label to the frame
        cv2.putText(
            frame, 
            emotions[maxindex], 
            (x+5, y-20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,  # Font size
            (255, 0, 0),  # Text color
            2,  # Thickness
            cv2.LINE_AA  # Anti-aliasing
        )

    # Display the processed video feed with emotion detection
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
