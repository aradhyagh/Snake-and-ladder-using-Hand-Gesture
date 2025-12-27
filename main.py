import cv2
import numpy as np
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound

# ================= USER SETTINGS =================
GRID_SIZE = 20
CELL_SIZE = 25
NUM_FRUITS = 5
NUM_WALLS = 30
CONTROL_MODE = "BOTH"   # KEYBOARD | INDEX | THUMB | BOTH
MODEL_PATH = r"C:\Users\User\Downloads\hand_landmarker.task"
# =================================================

WINDOW_SIZE = GRID_SIZE * CELL_SIZE
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)

speed = 2.5
MOVE_DELAY = int(500 // speed)

# ================= MEDIAPIPE =================
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# ================= UTILS =================
def is_opposite(d1, d2):
    return d1[0] == -d2[0] and d1[1] == -d2[1]

def reset_game():
    global snake, fruits, walls, score, paused, direction, game_over, last_move_time
    snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
    fruits = []
    walls = []
    score = 0
    paused = True
    direction = RIGHT
    game_over = False
    last_move_time = cv2.getTickCount()

    for _ in range(NUM_FRUITS):
        fruits.append(spawn_fruit())

    for _ in range(NUM_WALLS):
        walls.append(spawn_wall())

def set_direction(new_dir):
    global direction
    if len(snake) > 1 and is_opposite(direction, new_dir):
        return
    direction = new_dir

def spawn_fruit():
    while True:
        pos = (random.randint(0, GRID_SIZE - 1),
               random.randint(0, GRID_SIZE - 1))
        if pos not in snake and pos not in walls:
            return pos

def spawn_wall():
    while True:
        pos = (random.randint(0, GRID_SIZE - 1),
               random.randint(0, GRID_SIZE - 1))
        if pos not in snake and pos not in fruits and pos not in walls:
            return pos

# ================= HAND LOGIC =================
def get_direction_from_hand(landmarks, w, h):
    if paused or game_over:
        return

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    idx_tip, idx_pip = 8, 6
    mid_tip, mid_pip = 12, 10
    ring_tip, ring_pip = 16, 14
    pink_tip, pink_pip = 20, 18
    thumb_tip, thumb_ip = 4, 3

    index_up = pts[idx_tip][1] < pts[idx_pip][1]
    middle_dn = pts[mid_tip][1] > pts[mid_pip][1]
    ring_dn = pts[ring_tip][1] > pts[ring_pip][1]
    pinky_dn = pts[pink_tip][1] > pts[pink_pip][1]

    if CONTROL_MODE in ["INDEX", "BOTH"]:
        if index_up and middle_dn and ring_dn and pinky_dn:
            dx = pts[idx_tip][0] - pts[idx_pip][0]
            dy = pts[idx_tip][1] - pts[idx_pip][1]
            set_direction(RIGHT if abs(dx) > abs(dy) and dx > 0 else
                          LEFT if abs(dx) > abs(dy) else
                          DOWN if dy > 0 else UP)

    if CONTROL_MODE in ["THUMB", "BOTH"]:
        if middle_dn and ring_dn and pinky_dn:
            dx = pts[thumb_tip][0] - pts[thumb_ip][0]
            dy = pts[thumb_tip][1] - pts[thumb_ip][1]
            set_direction(RIGHT if abs(dx) > abs(dy) and dx > 0 else
                          LEFT if abs(dx) > abs(dy) else
                          DOWN if dy > 0 else UP)

# ================= DRAW GAME =================
def draw_game():
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

    for i in range(GRID_SIZE):
        cv2.line(img, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), (50, 50, 50), 1)
        cv2.line(img, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), (50, 50, 50), 1)

    for wx, wy in walls:
        cv2.rectangle(img, (wx * CELL_SIZE, wy * CELL_SIZE),
                      ((wx + 1) * CELL_SIZE, (wy + 1) * CELL_SIZE),
                      (255, 255, 255), -1)

    for i, (x, y) in enumerate(snake):
        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE),
                      ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE),
                      color, -1)

    for fx, fy in fruits:
        cv2.circle(img,
                   (fx * CELL_SIZE + CELL_SIZE // 2,
                    fy * CELL_SIZE + CELL_SIZE // 2),
                   CELL_SIZE // 3, (0, 0, 255), -1)

    cv2.putText(img, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if paused and not game_over:
        cv2.putText(img, "PAUSED",
                    (WINDOW_SIZE // 2 - 80, WINDOW_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if game_over:
        cv2.putText(img, "GAME OVER",
                    (WINDOW_SIZE // 2 - 120, WINDOW_SIZE // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(img, f"Final Score: {score}",
                    (WINDOW_SIZE // 2 - 120, WINDOW_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(img, "Press P to Play Again",
                    (WINDOW_SIZE // 2 - 160, WINDOW_SIZE // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(img, "Press Q to Quit",
                    (WINDOW_SIZE // 2 - 120, WINDOW_SIZE // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

# ================= MAIN =================
cap = cv2.VideoCapture(0)
timestamp = 0

score = 0
paused = True
game_over = False
direction = RIGHT
last_move_time = cv2.getTickCount()

reset_game()

print("Snake Game Running")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp += 15
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        get_direction_from_hand(result.hand_landmarks[0], w, h)
        for lm in result.hand_landmarks[0]:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 0), -1)

    key = cv2.waitKey(1) & 0xFF

    if game_over:
        if key == ord('p'):
            reset_game()
        elif key == ord('q'):
            break
    else:
        if key == ord('p'):
            paused = not paused
        elif key == ord('w'):
            set_direction(UP); paused = False
        elif key == ord('s'):
            set_direction(DOWN); paused = False
        elif key == ord('a'):
            set_direction(LEFT); paused = False
        elif key == ord('d'):
            set_direction(RIGHT); paused = False
        elif key == ord('q'):
            break

        current_time = cv2.getTickCount()
        elapsed = (current_time - last_move_time) / cv2.getTickFrequency() * 1000

        if elapsed > MOVE_DELAY and not paused:
            last_move_time = current_time
            head = snake[0]
            new_head = (head[0] + direction[0], head[1] + direction[1])

            if (new_head in snake or new_head in walls or
                    new_head[0] < 0 or new_head[1] < 0 or
                    new_head[0] >= GRID_SIZE or new_head[1] >= GRID_SIZE):
                winsound.PlaySound("crash.wav", winsound.SND_ASYNC)
                game_over = True

            else:
                snake.insert(0, new_head)
                if new_head in fruits:
                    fruits.remove(new_head)
                    fruits.append(spawn_fruit())
                    score += 1
                    winsound.PlaySound("bite.wav", winsound.SND_ASYNC)
                else:
                    snake.pop()

    cv2.imshow("Snake Game", draw_game())
    cv2.imshow("Webcam", cv2.resize(frame, None, fx=0.5, fy=0.5))

cap.release()
cv2.destroyAllWindows()
