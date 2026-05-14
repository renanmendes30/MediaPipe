```
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# MODELO
# =========================

model_path = "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# =========================
# VARIÁVEIS
# =========================

timestamp = 0
frames_fechado = 0

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(mp_image, timestamp)

    timestamp += 1

    h, w, _ = frame.shape

    if result.face_landmarks:

        face = result.face_landmarks[0]

        # =========================
        # LANDMARKS
        # =========================

        olho_esq = face[159]
        olho_dir = face[386]

        nariz = face[1]

        # =========================
        # POSIÇÕES
        # =========================

        x1 = int(olho_esq.x * w)
        y1 = int(olho_esq.y * h)

        x2 = int(olho_dir.x * w)
        y2 = int(olho_dir.y * h)

        xn = int(nariz.x * w)
        yn = int(nariz.y * h)

        # =========================
        # DESENHO
        # =========================

        cv2.circle(frame, (x1,y1), 5, (0,255,0), -1)
        cv2.circle(frame, (x2,y2), 5, (0,255,0), -1)

        cv2.circle(frame, (xn,yn), 5, (255,0,0), -1)

        # =========================
        # DETECÇÃO DE SONO
        # =========================

        olhos_fechados = olho_esq.y > 0.38 and olho_dir.y > 0.38

        cabeca_baixa = nariz.y > 0.6

        if olhos_fechados or cabeca_baixa:
            frames_fechado += 1
        else:
            frames_fechado = 0

        # =========================
        # ALERTA
        # =========================

        if frames_fechado > 15:

            cv2.putText(
                frame,
                "ALERTA! SONOLENCIA DETECTADA",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        else:

            cv2.putText(
                frame,
                "Motorista Atento",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

    cv2.imshow("Detector de Sono", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
