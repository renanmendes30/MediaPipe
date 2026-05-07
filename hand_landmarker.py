import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Modelo
model_path = "hand_landmarker.task"

# Configuração Tasks API
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# Vídeo / webcam
cap = cv2.VideoCapture(0)

timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 500))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Desenha pontos
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Dedo indicador (índice 8)
            index_finger_y = hand_landmarks[8].y

            # Lógica (igual seu código)
            if index_finger_y < 0.5:

                cv2.putText(frame, "90 graus", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
             
                cv2.putText(frame, "180 graus", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Control - Tasks API", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
