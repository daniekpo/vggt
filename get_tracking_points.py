import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
from typing import List
from pathlib import Path
import os

class PointSelector:
    """Interactive point selector for images and videos."""

    def __init__(self, input_path: str):
        """
        Initialize the point selector.

        Args:
            input_path: Path to image or video file
        """
        self.input_path = Path(input_path)
        self.points = []
        self.fig = None
        self.ax = None
        self.image = None
        self.original_image = None

        self.save_dir = os.path.dirname(input_path)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.image = self._load_image_or_first_frame()
        self.original_image = self.image.copy()
        self.img_name_no_ext = os.path.splitext(os.path.basename(input_path))[0]

    def _load_image_or_first_frame(self) -> np.ndarray:
        """Load image or extract first frame from video."""
        file_extension = self.input_path.suffix.lower()

        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self._load_image()
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            return self._extract_first_frame()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _load_image(self) -> np.ndarray:
        """Load image file."""
        image = cv2.imread(str(self.input_path))
        if image is None:
            raise ValueError(f"Could not load image: {self.input_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _extract_first_frame(self) -> np.ndarray:
        """Extract first frame from video."""
        cap = cv2.VideoCapture(str(self.input_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.input_path}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read first frame from video: {self.input_path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax or event.button != 1:  # Only left clicks in the axes
            return

        x, y = int(event.xdata), int(event.ydata)
        point_index = len(self.points)

        # Store the point
        self.points.append([x, y])

        # Draw the point on the image
        self._draw_point_on_image(x, y, point_index)

        # Update the display
        self.ax.imshow(self.image)
        self.fig.canvas.draw()

        print(f"Point {point_index}: ({x}, {y})")

    def _draw_point_on_image(self, x: int, y: int, index: int):
        """Draw a numbered circle at the specified point."""
        # Choose a unique color for each point
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 128),  # Teal
        ]

        color = colors[index % len(colors)]
        radius = max(10, min(self.image.shape[:2]) // 50)  # Adaptive radius based on image size

        # Draw filled circle
        cv2.circle(self.image, (x, y), radius, color, -1)

        # Draw black border
        cv2.circle(self.image, (x, y), radius, (0, 0, 0), 2)

        # Add number text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = radius / 20.0
        text_thickness = max(1, radius // 10)

        # Get text size to center it
        text = str(index)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

        # Center the text in the circle
        text_x = x - text_width // 2
        text_y = y + text_height // 2

        # Draw white text with black outline for better visibility
        cv2.putText(self.image, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness + 1)
        cv2.putText(self.image, text, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness)


    def _on_save(self, event):
        """Handle save button click."""
        if len(self.points) == 0:
            print("No points to save!")
            return

        with open(os.path.join(self.save_dir, f'{self.img_name_no_ext}_points.json'), 'w') as f:
            json.dump(self.points, f, indent=2)
        print(f"Saved {len(self.points)} points to {os.path.join(self.save_dir, f'{self.img_name_no_ext}_points.json')}")

        # Convert RGB back to BGR for OpenCV
        bgr_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.save_dir, f'{self.img_name_no_ext}_annotated.png'), bgr_image)
        print(f"Saved annotated image as {os.path.join(self.save_dir, f'{self.img_name_no_ext}_annotated.png')}")

    def _on_clear(self, event):
        """Handle clear button click."""
        self.points.clear()
        self.image = self.original_image.copy()
        self.ax.imshow(self.image)
        self.fig.canvas.draw()
        print("Cleared all points")

    def _setup_ui(self):
        """Setup the interactive UI."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title('Click on points to select them for tracking\nLeft click to add point, use buttons to save/clear')
        self.ax.axis('on')  # Show axes to help with coordinate reference

        # Add buttons
        ax_save = plt.axes([0.7, 0.01, 0.1, 0.05])
        ax_clear = plt.axes([0.81, 0.01, 0.1, 0.05])

        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear')

        # Connect button events
        self.btn_save.on_clicked(self._on_save)
        self.btn_clear.on_clicked(self._on_clear)

        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def run(self):
        """Run the interactive point selection interface."""
        self._setup_ui()

        print("Instructions:")
        print("- Left click on any point in the image to select it")
        print("- Points will be numbered and colored uniquely")
        print("- Click 'Save' to save points to points.json and image to annotated.png")
        print("- Click 'Clear' to remove all points")
        print("- Close the window when finished")

        plt.tight_layout()
        plt.show()


def select_tracking_points(input_path: str) -> List[List[int]]:
    """
    Main function to select tracking points from an image or video.

    Args:
        input_path: Path to the image or video file

    Returns:
        List of selected points as [x, y] coordinates
    """
    try:
        selector = PointSelector(input_path)
        selector.run()
        return selector.points
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python get_tracking_points.py <image_or_video_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    points = select_tracking_points(input_file)

    if points:
        print(f"\nSelected {len(points)} points:")
        for i, point in enumerate(points):
            print(f"Point {i}: {point}")
    else:
        print("No points were selected.")
