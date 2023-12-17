import cv2
import mediapipe as mp
import numpy as np
import asyncio
from PIL import Image

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

async def connect():
    # Read API key and API key ID from secrets.txt file
    with open('secrets.txt', 'r') as file:
        lines = file.readlines()
        api_key = lines[0].strip().split('=')[1]
        api_key_id = lines[1].strip().split('=')[1]

    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id
    )
    return await RobotClient.at_address('sickpi-main.acu4zxs3sy.viam.cloud', opts)

def is_visible(landmark):
    return landmark.visibility > 0.5 if hasattr(landmark, 'visibility') else True

def calculate_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def _calculate_scale_factor(landmarks):
    # Calculate the distance between the shoulders as the scale factor
    shoulder_distance = np.sqrt((landmarks['LEFT_SHOULDER'][0] - landmarks['RIGHT_SHOULDER'][0])**2 +
                                (landmarks['LEFT_SHOULDER'][1] - landmarks['RIGHT_SHOULDER'][1])**2)
    return shoulder_distance

def _frontal_view_score(landmarks, threshold):
    scale_factor = _calculate_scale_factor(landmarks)
    threshold = threshold * scale_factor
    score = 0

    # Check horizontal alignment of ears and eyes
    if abs(landmarks['LEFT_EYE'][1] - landmarks['RIGHT_EYE'][1]) < threshold:
        score += 1
    if abs(landmarks['LEFT_EAR'][1] - landmarks['RIGHT_EAR'][1]) < threshold:
        score += 1

    # Check distances between eyes and ears
    left_eye_ear_dist = abs(landmarks['LEFT_EYE'][0] - landmarks['LEFT_EAR'][0])
    right_eye_ear_dist = abs(landmarks['RIGHT_EYE'][0] - landmarks['RIGHT_EAR'][0])
    if abs(left_eye_ear_dist - right_eye_ear_dist) < threshold:
        score += 1
    elif left_eye_ear_dist < threshold or right_eye_ear_dist < threshold:
        score -= 1

    # Check shoulder alignment
    if abs(landmarks['LEFT_SHOULDER'][1] - landmarks['RIGHT_SHOULDER'][1]) < threshold:
        score += 1
    

    return score

def _side_view_score(landmarks, threshold):
    scale_factor = _calculate_scale_factor(landmarks)
    threshold = threshold * scale_factor
    score = 0

    # Check if one ear is significantly closer than the other
    if abs(landmarks['LEFT_EAR'][0] - landmarks['RIGHT_EAR'][0]) > threshold:
        score += 1

    # Check visibility of eyes
    if abs(landmarks['LEFT_EYE'][0] - landmarks['RIGHT_EYE'][0]) > threshold:
        score += 1
    

    # Increase score if eyes and ears are very close
    left_eye_ear_dist = abs(landmarks['LEFT_EYE'][0] - landmarks['LEFT_EAR'][0])
    right_eye_ear_dist = abs(landmarks['RIGHT_EYE'][0] - landmarks['RIGHT_EAR'][0])
    print(left_eye_ear_dist, right_eye_ear_dist, threshold)
    if left_eye_ear_dist < threshold or right_eye_ear_dist < threshold:
        score += 1

    # Check shoulder alignment
    if abs(landmarks['LEFT_SHOULDER'][0] - landmarks['RIGHT_SHOULDER'][0]) > threshold:
        score += 1

    return score

def determine_orientation(landmarks, threshold=0.10):
    frontal_score = _frontal_view_score(landmarks, threshold)
    side_score = _side_view_score(landmarks, threshold)
    print("Front Score: ", frontal_score, " ---- Side Score: ", side_score)

    if frontal_score > side_score:
        return "Frontal View"
    elif side_score > frontal_score:
        return "Side View"
    else:
        return "Undetermined"


async def main():
    robot = await connect()

    # Initialize mediapipe for pose detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # Get image from robot's webcam
            webcam = Camera.from_robot(robot, "webcam")
            webcam_return_value = await webcam.get_image()

            # Convert PIL Image to NumPy array
            frame = np.array(webcam_return_value)

            # Recolor image to RGB (if needed)
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR (if needed)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks_list = results.pose_landmarks.landmark

                    # Convert landmarks to a dictionary format using indices
                    landmark_dict = {}
                    for idx, landmark in enumerate(landmarks_list):
                        landmark_name = mp_pose.PoseLandmark(idx).name
                        landmark_dict[landmark_name] = [landmark.x, landmark.y]

                    orientation = determine_orientation(landmark_dict)

                    if orientation == "Frontal View":
                        print("Front View")
                        # Perform spine angle calculation for side view
                        # Add spine angle calculation code here
                        pass

                    elif orientation == "Side View":
                        print("Side View")
                        # Perform spine angle calculation for side view
                        # Add spine angle calculation code here
                        pass

                    else:
                        cv2.putText(image, "Unable to determine pose orientation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    print("No landmarks detected")

            except Exception as e:
                print(f"Error in pose estimation: {e}")

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
