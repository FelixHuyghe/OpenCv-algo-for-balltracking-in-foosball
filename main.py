#!/usr/bin/env python
# USAGE
# python main.py --video ball_tracking_example.mp4

# import the necessary packages
import sys
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math


# Python program for slope of line
def slope(x1, y1, x2, y2):
	if (x2 - x1 != 0):
		return (float)(y2 - y1) / (x2 - x1)
	return None


def intercept(s, point):
	return - (s * point[0]) + point[1]

# P1 and P2 describe a straight line
def distance_points_line(p1, p2, p3):
	p1 = np.asarray(p1)
	p2 = np.asarray(p2)
	p3 = np.asarray(p3)
	return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def calculate_intersection(line1, line2):
	m1 = slope(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
	m2 = slope(line2[0][0], line2[0][1], line2[1][0], line2[1][1])
	if m1 is None or m2 is None:
		return None
	c1 = intercept(m1, line1[0])
	c2 = intercept(m2, line2[0])


	x = np.linspace(-10, 10, 500)
	if abs((m2 - m1)) > 0.00005:
		xi = (c1 - c2) / (m2 - m1)
		yi = m1 * xi + c1
		return (xi, yi)
	else:
		return None


if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
					help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64,
					help="max buffer size")
	args = vars(ap.parse_args())

	lines = [[(0, 133), (73, 7)],
				 [(6, 326), (140, 6)],
				 [(122, 333), (207, 8)],
				 [(242, 335), (274, 4)],
				 [(362, 335), (343, 5)],
				 [(483, 335), (410, 3)],
				 [(598, 336), (475, 3)],
				 [(598, 127), (544, 13)]]

	# define the lower and upper boundaries of the "green"
	# ball in the HSV color space, then initialize the
	# list of tracked points
	greenLower = (0, 85, 150)
	greenUpper = (50, 255, 255)

	# greenLower = (29, 86, 6)
	# greenUpper = (64, 255, 255)

	pts = deque(maxlen=args["buffer"])

	# if a video path was not supplied, grab the reference
	# to the webcam
	if not args.get("video", False):
		vs = VideoStream(src=0).start()

	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(args["video"])

	# allow the camera or video file to warm up
	time.sleep(2.0)

	# keep looping
	count = 0

	while True:
		# grab the current frame
		frame = vs.read()

		# handle the frame from VideoCapture or VideoStream
		frame = frame[1] if args.get("video", False) else frame

		# if we are viewing a video and we did not grab a frame,
		# then we have reached the end of the video
		if frame is None:
			break

		# resize the frame, blur it, and convert it to the HSV
		# color space
		frame = imutils.resize(frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, greenLower, greenUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		center = None

		# loop over the set of tracked points
		for i in range(2, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 10)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 1:
				# cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2) # draw the circle and centroid on the frame,
				cv2.circle(frame, center, 3, (255, 0, 0), -1)  # then update the list of tracked points
		else:
			if len(pts) >= 3 and not pts[0] is None and not pts[1] is None and not pts[2] is None:
				nearestIntersect = 0
				nearestIntersectSet = False
				goingLeft = pts[0][0] < pts[2][0]
				curLine = [pts[0], pts[2]]
				for line in lines:
					dist = distance_points_line(line[0], line[1], pts[0])
					if dist < 20:
						pointRight = (pts[0][0] + dist, pts[0][1])
						pointLeft = (pts[0][0] - dist, pts[0][1])
						if distance_points_line(line[0], line[1], pointLeft) < distance_points_line(line[0], line[1],
																									pointRight):
							nearestIntersect = pointLeft
						else:
							nearestIntersect = pointRight
						nearestIntersectSet = True
						break
					intersect = calculate_intersection(line, curLine)
					if intersect is not None:
						if not nearestIntersectSet:
							if goingLeft and pts[0][0] > intersect[0]:
								nearestIntersect = intersect
								nearestIntersectSet = True
							elif not goingLeft and pts[0][0] < intersect[0]:
								nearestIntersect = intersect
								nearestIntersectSet = True
						else:
							if goingLeft:
								if pts[0][0] > intersect[0] > nearestIntersect[0]:
									nearestIntersect = intersect
							else:
								if pts[0][0] < intersect[0] < nearestIntersect[0]:
									nearestIntersect = intersect

				if nearestIntersectSet:
					center = (int(nearestIntersect[0]), int(nearestIntersect[1]))
					cv2.circle(frame, center, 3, (255, 255, 0), -1)  # then update the list of tracked points


		# update the points queue
		pts.appendleft(center)

		# show the frame to our screen
		cv2.imshow("Frame", frame)

		name = "generated_images/frame%d.jpg" % count
		# Uncomment this to save the frames
		cv2.imwrite(name, frame)  # save frame as JPEG file
		count += 1

		key = cv2.waitKey(1) & 0xFF

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

	# if we are not using a video file, stop the camera video stream
	if not args.get("video", False):
		vs.stop()

	# otherwise, release the camera
	else:
		vs.release()

	# close all windows
	cv2.destroyAllWindows()
