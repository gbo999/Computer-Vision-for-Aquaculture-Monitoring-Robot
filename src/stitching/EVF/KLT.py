import cv2
import numpy as np

# Load the video
video_path = 'path/to/your/video.mp4'  # Update this with the path to your video
cap = cv2.VideoCapture(video_path)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a list to hold the trajectories
trajectories = [np.float32(p0)]

# Variable to hold the segments
segments = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e., track feature points)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Filter out the points with high error
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        good_err = err[st == 1]
        p1 = good_new[good_err < 12]  # Threshold for error

        # Update the trajectories
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            if i < len(trajectories):
                trajectories[i] = np.vstack([trajectories[i], new])

    # Check if we need to start a new segment
    if len(trajectories[-1]) > 30:  # This is an arbitrary threshold
        # Check the consistency of the trajectories
        displacement = np.linalg.norm(trajectories[-1][-1] - trajectories[-1][0])
        if displacement > 50:  # This is an arbitrary threshold for "significant" movement
            segments.append(trajectories[-1])
            trajectories = [np.float32(p1)]

    # Update the previous frame and previous points for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Add the last segment if it wasn't added
if trajectories and not any(np.array_equal(trajectories[-1], segment) for segment in segments):
    segments.append(trajectories[-1])

# Release the video capture
cap.release()

# Process segments as needed
print(f"Detected {len(segments)} segments based on feature point trajectories.")
# You can now process these segments as needed for your application
