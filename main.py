from src.video_processor import *

processor = Pipeline()
img = cv2.imread('test_images/test4.jpg')
results = processor.process(img)
# results = processor.threshold(img)
show_four_images(results)

# v_processor = VideoProcessor()
# # v_processor.process_video('challenge_video.mp4')
# v_processor.process_video()
