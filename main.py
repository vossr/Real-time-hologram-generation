from depth import depth_estimation
import display_cap
import numpy as np
import cv2

window_title = "depth"
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
		cv2.imshow(window_title, combined)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()

def screen_realtime():
	cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	while True:
		frame = display_cap.get_rect()
		frame = np.array(frame)

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
