import cv2
import logging
import logging.config

from utils import detections, CentroidTracker


# setup logger
logging.config.fileConfig("log/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__file__)


def display(frame, tracker, bboxes):
    canvas = frame.copy()
    for person in tracker.objects.values():
        text = f"ID: {person.id_}"
        cx, cy = person.centroid.astype("int")
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            canvas, text, (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return canvas


def track_video(vcap):
    tracker = CentroidTracker()
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
            frame = display(frame, tracker, bboxes)

        cv2.imshow("", frame)
    cv2.destroyAllWindows()
    vcap.release()


if __name__ == "__main__":
    track_video(cv2.VideoCapture(2))
