from pipeline import *
from utility import *

processor = Pipeline()
img = cv2.imread('test_images/test5.jpg')
results = processor.process(img)
show_four_images(results)
