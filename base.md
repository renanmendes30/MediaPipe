### Código base simplificado:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        for landmark in result.hand_landmarks[0]:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    cv2.imshow("Hand", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

##  Identificar dedo indicador 

* Landmark 8 = ponta do dedo indicador

### Código:

```python
index = result.hand_landmarks[0][8]
h, w, _ = frame.shape
x, y = int(index.x * w), int(index.y * h)

cv2.circle(frame, (x, y), 10, (255,0,0), -1)
```

---

Estados do dedo (30 min)

### Objetivo:

Criar lógica de posição vertical

```python
if index.y < 0.3:
    texto = "ACIMA"
elif index.y < 0.6:
    texto = "MEIO"
else:
    texto = "ABAIXO"

cv2.putText(frame, texto, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
```

---

##  Controle de objeto 

### Objetivo:

Mover objeto com o dedo

```python
cv2.circle(frame, (x, y), 20, (0,0,255), -1)
```

---

### Criar barra de "volume":

```python
altura_barra = int((1 - index.y) * 300)
cv2.rectangle(frame, (50, 400), (100, 400 - altura_barra), (0,255,0), -1)
```

