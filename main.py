import argparse
from functions.camera_stream import CameraStream
from model.computer_vision import ImageProcessor

class MainApplication:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.camera_stream = CameraStream()

    def main(self):
        parser = argparse.ArgumentParser(description="Image processing script")
        parser.add_argument('--download', action='store_true', help='Download images')
        parser.add_argument('--train', action='store_true', help='Train the model')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
        parser.add_argument('--camera', action='store_true', help='Use camera')

        args = parser.parse_args()

        if args.download:
            self.image_processor.download_images()

        if args.train:
            self.image_processor.train_model()

        if args.evaluate:
            self.image_processor.evaluate_model()

        if args.camera:
            self.camera_stream.show_frame()

if __name__ == "__main__":
    app = MainApplication()
    app.main()
