#Libraries
import cv2
import numpy as np
import dlib
import time

#text_format
font = cv2.FONT_HERSHEY_PLAIN
#Camera_access
cap = cv2.VideoCapture(0)
#model that used dlib library to detect the face
detector = dlib.get_frontal_face_detector()
#to detect the face landmarks
#predictor = dlib.shape_predictor(r"C:\Users\Padmashree\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(r"C:\Users\Padmashree\Desktop\Australia\MASTER IN AI\Semester 4\FIT5120\IE Project\Code\Iteration-2\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
#array to store state change counts in binary form
state_changes_binary = []
#variables to count frames in attended and distracted states
frames_attended = 0
frames_distracted = 0
total_frames = 0
start_time = time.time()

#function to gaze the detected eye
def get_gaze_ratio(eye_points, facial_landmarks, gray, frame):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                 (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                 (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                 (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                 (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                 (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    #print("left_side_white",left_side_white)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    #print("right_side_white",right_side_white)
    if left_side_white == 0:
        gaze_ratio = 0.7
    elif right_side_white == 0:
        gaze_ratio = 1.6
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        #Eye tracking part
        if gaze_ratio <= 0.7:
            cv2.putText(frame, "DISTRACTED", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
            # Add 1 to the array to indicate transition from center to distracted
            state_changes_binary.append(1)
            frames_distracted += 1                 #right
        elif 0.7 < gaze_ratio < 1.6:
            cv2.putText(frame, "ATTENDED", (50, 100), font, 2, (0, 0, 255), 3)
            # Add 0 to the array to indicate no transition
            state_changes_binary.append(0)
            frames_attended += 1
        else:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "DISTRACTED", (50, 100), font, 2, (0, 0, 255), 3)
            # Add 1 to the array to indicate transition from center to distracted
            state_changes_binary.append(1)
            frames_distracted += 1
        total_frames += 1                          #left
    
    # Calculate and print accuracy of being attended

    
    cv2.imshow("Frame", frame)
    cv2.imshow("New frame", new_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the array after the loop
print("State changes in binary form:", state_changes_binary)
accuracy = (frames_attended / total_frames) * 100 if total_frames > 0 else 0
print(f"Accuracy of being attended: {accuracy:.2f}%")
print(f"Frames in attended state: {frames_attended}")
print(f"Frames in distracted state: {frames_distracted}")

# Print total time spent in the video stream
end_time = time.time()
total_time = end_time - start_time
print(f"Total time spent: {total_time:.2f} seconds")
