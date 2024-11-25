import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO('best_v2.pt')  # Update with the path to your trained model

def detect_and_display():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if not file_path:
        return  # If no file is selected, do nothing

    # Run YOLO detection on the selected image
    results = model.predict(source=file_path, save=False)  # Get the predictions
    result = results[0]  # Access the first (and only) result

    # Convert the result image to a format compatible with Tkinter
    img = result.plot()  # Annotate the image with YOLO detections
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = Image.fromarray(img)  # Convert to PIL Image

    # Automatically adjust window size to fit the image
    window_width, window_height = img.size
    root.geometry(f"{window_width}x{window_height+50}")  # Add extra space for the button

    # Display the image in the Tkinter window
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

# Create the main Tkinter window
root = tk.Tk()
root.title("YOLOv8 Detection")

# Create a button to trigger image selection and detection
btn = tk.Button(root, text="Select Image", command=detect_and_display)
btn.pack(pady=10)

# Create a label to display the images
panel = tk.Label(root)
panel.pack()

# Start the Tkinter main loop
root.mainloop()
