import cv2
import mediapipe as mp

# Import da nova API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Caminho do modelo
model_path = r"pose_landmarker.task"

# Configuração do modelo
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Cria o detector
detector = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converte para formato MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Processa (IMPORTANTE: precisa timestamp)
    result = detector.detect_for_video(mp_image, timestamp)

    timestamp += 1

    # Verifica se encontrou pose
    if result.pose_landmarks:
        for landmark in result.pose_landmarks[0]:
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.imshow("Pose - Tasks API", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()