from utils import read_video, save_video
from trackers import Tracker
from trackers import KeyPoints
from mini_pitch import MiniPitch
from assign_teams import Teams
from camera_movement import CameraMovementEstimator

def main():
    INPUT_VIDEO_NAME = "inference_2.mp4"
    OUTPUT_VIDEO_NAME = "output_video_2"

    input_video_dir = f"Videos/{INPUT_VIDEO_NAME}"
    output_video_dir = f"output_videos/{OUTPUT_VIDEO_NAME}.avi"
    
    video_frames = read_video(input_video_dir)
    pitch = MiniPitch(video_frames)
   

    tracker = Tracker("models/player_detection.pt")
    key_points = KeyPoints("models/pitch_keypointsv3.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stubs=True,
                                       stub_path=f'stubs/{INPUT_VIDEO_NAME}_tracks.pkl')
     
    points = key_points.detect_frames(video_frames,
                                      read_from_stubs=True,
                                      stub_path=f'stubs/{INPUT_VIDEO_NAME}_points.pkl')

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                             read_from_stubs=True,
                                                                             stub_path=f'stubs/{INPUT_VIDEO_NAME}_camera_movement.pkl')
    
    
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    teams = Teams()
    teams.assign_color_to_team(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team =  teams.get_player_team(video_frames[frame_num],
                                          track['bbox'],
                                          player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = teams.team_colors[team]



    player_mini_court_detections, ball_mini_court_detections = pitch.project_court_to_mini_court(tracks,points,video_frames)
    

    output_frames = tracker.draw_annotations(video_frames,tracks)
    output_frames_1 = camera_movement_estimator.draw_camera_movement(output_frames,camera_movement_per_frame)
    output_frames_2 = key_points.draw_annotations(output_frames_1, points)
    output_frames_3 = pitch.draw_mini_pitch(output_frames_2)
    output_frames_4 = pitch.draw_points_on_minipitch(output_frames_3,player_mini_court_detections)
    output_frames_5 = pitch.draw_points_on_minipitch(output_frames_4,ball_mini_court_detections)
    

    save_video(output_frames_5, output_video_dir)

if __name__ == "__main__":
    main() 