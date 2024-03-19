from flask import Flask, request, url_for, redirect, render_template, Response
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

shrey_image = face_recognition.load_image_file("Shrey/shrey.jpg")
shrey_face_encoding = face_recognition.face_encodings(shrey_image)[0]

shreyv2_image = face_recognition.load_image_file("Shrey/shreyv2.jpg")
shreyv2_face_encoding = face_recognition.face_encodings(shreyv2_image)[0]

known_face_encodings = [
    shrey_face_encoding,
    shreyv2_face_encoding
]

known_face_names = [
    "Shrey",
    "Shreyv2"
]

def gen_frames():
    """
    Generator function that reads frames from a camera and detects faces and eyes in each frame.

    Yields:
        bytes: JPEG image frames as multipart response.
    """
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # # Initialize face and eye detectors
            # detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            # eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

            # # Detect faces in the frame
            # # Use the detectMultiScale method of the detector object to detect faces in the frame.
            # # The method returns a list of rectangles representing the detected faces.
            # faces = detector.detectMultiScale(frame, 1.1, 7)

            # # Convert the frame from the BGR color space to grayscale.
            # # This is often done before performing certain image processing tasks, as it simplifies the computation.
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # # Iterate over each face rectangle in the faces list.
            # for (x, y, w, h) in faces:
            #     # Draw a rectangle around the detected face on the frame.
            #     # The rectangle is defined by its top-left corner (x, y), its width w, and its height h.
            #     # The rectangle is drawn in blue color (255, 0, 0) and with a thickness of 2 pixels.
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #     # Extract the region of interest (ROI) from the grayscale image.
            #     # The ROI corresponds to the detected face rectangle.
            #     roi_gray = gray[y:y+h, x:x+w]

            #     # Extract the ROI from the color image.
            #     # The ROI corresponds to the detected face rectangle.
            #     roi_color = frame[y:y+h, x:x+w]

            #     # Use the detectMultiScale method of the eye_cascade object to detect eyes in the ROI.
            #     # The method returns a list of rectangles representing the detected eyes.
            #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            #     # Iterate over each eye rectangle in the eyes list.
            #     for (ex, ey, ew, eh) in eyes:
            #         # Draw a rectangle around the detected eye on the ROI.
            #         # The rectangle is defined by its top-left corner (ex, ey), its width ew, and its height eh.
            #         # The rectangle is drawn in green color (0, 255, 0) and with a thickness of 2 pixels.
            #         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            #small_frame = cv2.resize(frame, (0, 0), fx=0.125, fy=0.125)

            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                #face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                else:
                    print("No faces found in the image.")
                    break

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Encode the frame as JPEG image    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as multipart response
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()