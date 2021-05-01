from moviepy.editor import VideoFileClip
from src.pipeline import *


class VideoProcessor:
    def __init__(self):
        self.image_pipeline = Pipeline()
        self.count = 1

    def process_image(self, img, plot_output=False):
        # pipeline is processing BGR image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        undistorted, mask, warped, fitted, result = self.image_pipeline.process(img)
        if plot_output:
            if self.count % 30 == 0:
                show_four_images((undistorted, mask, warped, fitted))
                self.count = 1
            else:
                self.count = self.count + 1
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def process_video(self, video_name='project_video.mp4'):
        output = 'output_images/' + video_name
        clip = VideoFileClip(video_name)
        new_clip = clip.fl_image(self.process_image)
        new_clip.write_videofile(output, audio=False)
