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

```
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIGURAÇÃO DO MODELO
# =========================

model_path = "pose_landmarker.task"

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)

# =========================
# VARIÁVEIS
# =========================

contador = 0
estado_aberto = False
timestamp = 0

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(1)

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

    if result.pose_landmarks:

        pose = result.pose_landmarks[0]

        # =========================
        # LANDMARKS
        # =========================

        mao_esq = pose[15]
        mao_dir = pose[16]

        pe_esq = pose[31]
        pe_dir = pose[32]

        # =========================
        # POSIÇÕES
        # =========================

        mx1, my1 = int(mao_esq.x * w), int(mao_esq.y * h)
        mx2, my2 = int(mao_dir.x * w), int(mao_dir.y * h)

        px1, py1 = int(pe_esq.x * w), int(pe_esq.y * h)
        px2, py2 = int(pe_dir.x * w), int(pe_dir.y * h)

        # =========================
        # DESENHO
        # =========================

        cv2.circle(frame, (mx1, my1), 10, (255,0,0), -1)
        cv2.circle(frame, (mx2, my2), 10, (255,0,0), -1)

        cv2.circle(frame, (px1, py1), 10, (0,255,0), -1)
        cv2.circle(frame, (px2, py2), 10, (0,255,0), -1)

        # =========================
        # LÓGICA DO POLICHINELO
        # =========================

        maos_cima = mao_esq.y < 0.4 and mao_dir.y < 0.4

        pernas_abertas = abs(pe_esq.x - pe_dir.x) > 0.3

        # posição aberta
        if maos_cima and pernas_abertas and not estado_aberto:
            estado_aberto = True

        # voltou para posição inicial
        if not maos_cima and not pernas_abertas and estado_aberto:
            contador += 1
            estado_aberto = False

        # =========================
        # TEXTO
        # =========================

        cv2.putText(
            frame,
            f"Polichinelos: {contador}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    cv2.imshow("Contador de Polichinelos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
