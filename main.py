from pipeline import *
from utility import *

processor = Pipeline()
img = cv2.imread('test_images/test5.jpg')
results = processor.process(img)
# results = processor.threshold(img)
show_four_images(results)
