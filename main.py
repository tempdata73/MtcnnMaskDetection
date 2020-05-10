import cv2
import logging
import logging.config

import utils

from utils import detections, CentroidTracker


# setup logger
logging.config.fileConfig("log/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__file__)


def display(frame, tracker, bboxes, scores):
    canvas = frame.copy()
    canvas = utils.draw_tracker(canvas, tracker)
    canvas = utils.draw_bounding_boxes(canvas, bboxes, scores)
    return canvas


def track_video(vcap):
    tracker = CentroidTracker(max_detections=10)
    while True:
        ret, frame = vcap.read()
        key = cv2.waitKey(50)
        if not ret or key == ord("q"):
            logger.info("Stopped video processing")
            break

        # do face tracking and ignore facial landmarks
        bboxes, scores = detections.fetch_faces(frame, return_landmarks=False)
        # update face tracker
        centroids = detections.fetch_centroids(bboxes)
        tracker.update(centroids)

        # show results
        if len(centroids) > 0:
            frame = display(frame, tracker, bboxes, scores)

        cv2.imshow("", frame)
    cv2.destroyAllWindows()
    vcap.release()


if __name__ == "__main__":
    track_video(cv2.VideoCapture(2))
