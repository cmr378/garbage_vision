import cv2
import time

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera with ID {self.camera_id}")
        self.prev_time = time.time()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        return frame

    def show_frame(self, window_name='Camera'):
        while True:
            frame = self.get_frame()
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            # Display FPS on the frame
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Print FPS to the console
            print(f"FPS: {fps:.2f}")

            cv2.imshow(window_name, frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_camera()

    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = CameraStream()
    cam.show_frame()
