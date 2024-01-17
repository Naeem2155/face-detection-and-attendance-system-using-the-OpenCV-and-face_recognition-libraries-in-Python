import cv2
import face_recognition
import datetime
import numpy as np

# Load the training data
known_face_encodings = []
known_face_names = []

# Load the image of the person to be marked as present
current_student_image = face_recognition.load_image_file("current_student.jpg")
current_student_image = cv2.cvtColor(current_student_image, cv2.COLOR_RGB2BGR)

# Encode the current student's face
current_student_encoding = face_recognition.face_encodings(current_student_image)[0]

# Load the video capture device (webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over all detected faces in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face encoding with the known face encodings
        results = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If there is a match, find the index of the matched face
        if True in results:
            match_index = results.index(True)
            name = known_face_names[match_index]

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name of the person above the box
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # If the current student is detected, mark attendance
        if name == "Current Student":
            markAttendance(current_student_image, current_student_encoding, known_face_names)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the windows
video_capture.release()
cv2.destroyAllWindows()

def markAttendance(current_student_image, current_student_encoding, known_face_names):
    # Encode the current student's face
    current_student_encoding = face_recognition.face_encodings(current_student_image)[0]

    # Compare the current student's face encoding with the known face encodings
    results = face_recognition.compare_faces(known_face_encodings, current_student_encoding)

    # If there is a match, find the index of the matched face
    if True in results:
        match_index = results.index(True)

        # If the current student is already in the list, don't add them again
        if known_face_names[match_index] == "Current Student":
            return

        # Add the current student to the list of known faces
        known_face_encodings.append(current_student_encoding)
        known_face_names.append("Current Student")

        # Save the updated lists to disk
        with open("known_face_encodings.pickle", "wb") as f:
            pickle.dump(known_face_encodings, f)

        with open("known_face_names.pickle", "wb") as f:
            pickle.dump(known_face_names, f)

        # Mark attendance
        now = datetime.datetime.now()
        attendance_file = open("attendance.csv", "a")
        attendance_file.write(f"{now},{known_face_names[match_index]}\n")
        attendance_file.close()