# Import OpenCV library for computer vision tasks like image and video processing.
import cv2

# Import mediapipe library for utilizing its pre-built solutions for tasks like hand tracking, pose estimation, etc.
import mediapipe as mp

# Import the hypot function from the math module to calculate the Euclidean distance between two points.
from math import hypot

# Import NumPy library for numerical computations and operations on arrays and matrices.
import numpy as np

# Import screen_brightness_control library to control the brightness of the screen programmatically.
import screen_brightness_control as sbc

# Import necessary modules for controlling audio volume using Windows Core Audio API (pycaw library).
from ctypes import cast, POINTER, HRESULT
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from comtypes import CLSCTX_ALL

def get_left_right_landmarks(frame, processed, draw, mpHands):
    left_landmarkList = []
    right_landmarkList = []

    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            for idx, landmarks in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                if idx == 4 or idx == 8:
                    landmark = [idx, x, y]
                    if handlm == processed.multi_hand_landmarks[0]:
                        left_landmarkList.append(landmark)
                    elif handlm == processed.multi_hand_landmarks[1]:
                        right_landmarkList.append(landmark)

            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return left_landmarkList, right_landmarkList

"""## **Function: get_distance**

This function calculates the distance between two landmarks in a list and visualizes them on the frame.

Parameters:
- `frame`: The current frame from the video feed.
- `landmark_list`: List of landmarks (containing at least two landmarks).

1. `if len(landmark_list) < 2:`: Checks if there are at least two landmarks in the list.
    - `return`: Exits the function if there are not enough landmarks.

2. `(x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), (landmark_list[1][1], landmark_list[1][2])`: Extracts the coordinates of two landmarks from the `landmark_list`.

3. Draws circles and a line to visualize the detected landmarks on the frame using OpenCV:
   - `cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)`
   - `cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)`
   - `cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)`

4. Calculates the [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between the two landmarks using the `hypot` function from the `math` module:
   - `L = hypot(x2 - x1, y2 - y1)`

5. Returns the calculated distance (`L`).

Overall, this function provides the distance between two landmarks in pixels and visually represents them on the frame for debugging and visualization purposes.

"""

def get_distance(frame, landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1, y1), (x2, y2) = (landmark_ist[0][1], landmark_ist[0][2]), \
        (landmark_ist[1][1], landmark_ist[1][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    L = hypot(x2 - x1, y2 - y1)

    return L


def start_magic():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            left_landmarkList, right_landmarkList = get_left_right_landmarks(frame, processed, draw, mpHands)

            #Change brightness using left hand (In video it would appear as right hand as we are mirroring the frame)
            if left_landmarkList:
                left_distance = get_distance(frame, left_landmarkList)
                b_level = np.interp(left_distance, [50, 220], [0, 100])
                sbc.set_brightness(int(b_level))

            # Change volume using right hand (In video it would appear as left hand as we are mirroring the frame)
            if right_landmarkList:
                right_distance = get_distance(frame, right_landmarkList)
                vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

