# import cv2    
# import time
# cpt = 0
# maxFrames = 10000 # if you want 5 frames only.


# cap=cv2.VideoCapture(r"C:\Users\Angelo\Desktop\Project\Videos\Liverpool vs Fulham Highlights.ts")
# while cpt < maxFrames:
#     ret, frame = cap.read()
#     frame=cv2.resize(frame,(848,480))
#     time.sleep(0.01)
#     frame=cv2.flip(frame,1)
#     cv2.imshow("test window", frame) # show image in window
#     cv2.imwrite(r"C:\Users\Angelo\Desktop\Project\Stock Images" %cpt, frame)
#     cpt += 1
#     if cv2.waitKey(5)&0xFF==27:
#         break
# cap.release()   
# cv2.destroyAllWindows()

import cv2
import os

def extract_frames(video_path, output_dir, output_resolution,frame_interval=1):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame, output_resolution)
        # Save frame if it's at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{'game_3'}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {frame_count}")

        frame_count += 1

    cap.release()

# Example usage
video_path = r"C:\Users\Angelo\Desktop\Project\Videos\UEFA Europan Qualifiers Highlights Show - 12 September 2023 - Source 2 - FootballOrgin.ts"
output_dir = r"C:\Users\Angelo\Desktop\Project\Stock Images"
frame_interval = 25  # Adjust this interval as needed
output_resolution = (848, 480)
extract_frames(video_path, output_dir, output_resolution ,frame_interval)
