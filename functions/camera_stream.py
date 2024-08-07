import cv2

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera with ID {self.camera_id}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        return frame

    def show_frame(self, window_name='Camera'):
        while True:
            frame = self.get_frame()
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_camera()

    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = CameraStream()
    cam.show_frame()
