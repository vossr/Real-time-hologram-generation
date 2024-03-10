from depth import depth_estimation
from mss import mss
import numpy as np
import cv2

sct = mss()
monitor_number = 2
monitor = sct.monitors[monitor_number]



def video_viewer(video_path):
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("Error opening video file")

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		frame = cv2.resize(frame, (0, 0), fx=0.70, fy=0.70)
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		depth = depth_estimation(frame)

		combined = cv2.hconcat([frame, depth])
		cv2.imshow('depth', combined)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()

def crop_portrait_center(image):
	original_height, original_width, channels = image.shape
	new_width = original_width // 2
	x_start = (original_width - new_width) // 2
	x_end = x_start + new_width
	cropped_image = image[:, x_start:x_end, :]
	return cropped_image

def crop_more(image):
	height, width, channels = image.shape
	cutt = int(0.1 * height)
	cutb = height - int(0.3 * height)
	cutl = int(width * 0.2)
	cutr = width - int(width * 0.2)
	image = image[cutt:cutb, cutl:cutr]
	return image

def screen_realtime():
	window_title = "depth"
	cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	while True:
		frame = sct.grab(monitor)
		frame = np.array(frame)

		frame = crop_portrait_center(frame)
		frame = crop_more(frame)

		# drop alpha channel
		if frame.shape[2] == 4:
			frame = frame[:, :, :3]

		depth = depth_estimation(frame)
		combined = cv2.hconcat([frame, depth])

		cv2.imshow(window_title, combined)
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()


# video_viewer('vid.mp4')
screen_realtime()
