import cv2

def read_video(video_path):
    """
    Function collects the video frames to be used for further processing
    video_path (path): path of the input video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames, output_dir):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_dir, fourcc, 24, (output_frames[0].shape[1],output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()
