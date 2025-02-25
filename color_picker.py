import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import sys
import os
import io
import pyautogui
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas

MAX_BLOBS = 100  # Set a limit for the maximum number of blobs

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        self.widget.insert(tk.END, string)
        self.widget.see(tk.END)  # Auto-scroll to the end

    def flush(self):  # Added to handle flush behavior (no-op)
        pass

def load_action():
    global loaded_image, resized_image, image_zoom_label
    
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if file_path:
        loaded_image = Image.open(file_path)
        print(f"Image loaded: {file_path}")
        update_zoom_image(zoom_slider.get())

def update_zoom_image(zoom_level):
    global loaded_image, resized_image, image_zoom_canvas, mask_canvas, blob_canvas, color_canvas, current_zoom_level, visible_region
    
    if loaded_image is not None:
        current_zoom_level = zoom_level
        # Calculate the new size based on zoom level
        scale_factor = zoom_level / 50.0  # Zoom level: 50 means no zoom (100%)
        new_width = int(loaded_image.width * scale_factor)
        new_height = int(loaded_image.height * scale_factor)

        resized_image = loaded_image.resize((new_width, new_height), Image.LANCZOS)

        # Determine the visible region
        canvas_width = image_zoom_canvas.winfo_width()
        canvas_height = image_zoom_canvas.winfo_height()
        x_offset = max((new_width - canvas_width) // 2, 0)
        y_offset = max((new_height - canvas_height) // 2, 0)
        visible_region = resized_image.crop((x_offset, y_offset, x_offset + canvas_width, y_offset + canvas_height))

        # Clear the canvas
        image_zoom_canvas.delete("all")

        # Create a new PhotoImage
        tk_image = ImageTk.PhotoImage(resized_image)

        # Center the image on the canvas
        x_offset_canvas = (canvas_width - new_width) // 2
        y_offset_canvas = (canvas_height - new_height) // 2

        image_zoom_canvas.create_image(x_offset_canvas, y_offset_canvas, anchor=tk.NW, image=tk_image)
        image_zoom_canvas.image = tk_image

        # Update the mask, blob, and color views to match the zoomed image
        update_mask_view()
        update_blob_view()
        update_color_view()

def update_mask_view():
    global resized_image, mask_canvas, hue_lower_slider, hue_upper_slider, sat_lower_slider, sat_upper_slider, bri_lower_slider, bri_upper_slider, current_mask, visible_region
    
    if resized_image is not None and visible_region is not None:
        # Convert the visible region to HSV
        hsv_image = visible_region.convert("HSV")
        hsv_array = np.array(hsv_image)

        # Extract HSV components
        hue, saturation, brightness = hsv_array[:,:,0], hsv_array[:,:,1], hsv_array[:,:,2]

        # Get the lower and upper bound values from sliders
        hue_lower = hue_lower_slider.get()
        hue_upper = hue_upper_slider.get()
        sat_lower = sat_lower_slider.get()
        sat_upper = sat_upper_slider.get()
        bri_lower = bri_lower_slider.get()
        bri_upper = bri_upper_slider.get()

        # Create a binary mask for the visible region within the specified bounds
        mask = ((hue >= hue_lower) & (hue <= hue_upper) &
                (saturation >= sat_lower) & (saturation <= sat_upper) &
                (brightness >= bri_lower) & (brightness <= bri_upper))
        mask = mask.astype(np.uint8) * 255
        current_mask = mask  # Store the current mask for blob extraction

        # Convert the mask to an image
        mask_image = Image.fromarray(mask)

        # Clear the mask canvas
        mask_canvas.delete("all")

        # Create a new PhotoImage for the mask
        mask_tk_image = ImageTk.PhotoImage(mask_image)

        # Center the mask image on the canvas
        canvas_width = mask_canvas.winfo_width()
        canvas_height = mask_canvas.winfo_height()
        new_width, new_height = mask_image.size
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        mask_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=mask_tk_image)
        mask_canvas.image = mask_tk_image

def update_blob_view():
    global resized_image, blob_canvas, color_canvas, current_mask, largest_blob_rgb
    
    if resized_image is not None and current_mask is not None:
        # Find connected components (blobs) in the mask
        num_labels, labels_im = cv2.connectedComponents(current_mask)

        # Update blob count
        blob_count_label.config(text=f"Blob Count: {num_labels - 1}")

        # Handle cases with no blobs or too many blobs
        if num_labels == 1:
            handle_no_blobs()
            return
        if num_labels > MAX_BLOBS:
            handle_too_many_blobs()
            return

        # Find the largest component (excluding the background label 0)
        largest_blob_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])

        # Create a mask for the largest blob
        largest_blob_mask = (labels_im == largest_blob_label).astype(np.uint8) * 255

        # Create a white background image
        white_background = Image.new("RGB", visible_region.size, (255, 255, 255))

        # Convert the visible region to RGB
        visible_rgb = visible_region.convert("RGB")

        # Apply the largest blob mask to the visible region
        mask = Image.fromarray(largest_blob_mask).convert("L")
        blob_image = Image.composite(visible_rgb, white_background, mask)

        # Clear the blob canvas
        blob_canvas.delete("all")

        # Create a new PhotoImage for the blob
        blob_tk_image = ImageTk.PhotoImage(blob_image)

        # Center the blob image on the canvas
        canvas_width = blob_canvas.winfo_width()
        canvas_height = blob_canvas.winfo_height()
        new_width, new_height = blob_image.size
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        blob_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=blob_tk_image)
        blob_canvas.image = blob_tk_image

        # Calculate the average RGB value of the largest blob
        largest_blob_pixels = np.array(visible_rgb)[labels_im == largest_blob_label]
        largest_blob_rgb = largest_blob_pixels.mean(axis=0)

        # Update the color view with the average RGB value
        update_color_view()

        # Update the RGB value label
        avg_rgb_label.config(text=f"Avg RGB: {tuple(map(int, largest_blob_rgb))}")

        # Update the largest blob size
        largest_blob_size_label.config(text=f"Largest Blob Size: {len(largest_blob_pixels)}")

        # Update the largest blob coordinates
        largest_blob_coords = np.column_stack(np.where(labels_im == largest_blob_label))
        largest_blob_coords_label.config(text=f"Largest Blob Coordinates: {largest_blob_coords[0]}")

def update_color_view():
    global color_canvas, largest_blob_rgb
    
    if largest_blob_rgb is not None:
        # Create a solid color image using the average RGB value
        color_image = Image.new("RGB", (100, 100), tuple(map(int, largest_blob_rgb)))

        # Clear the color canvas
        color_canvas.delete("all")

        # Create a new PhotoImage for the color
        color_tk_image = ImageTk.PhotoImage(color_image)

        # Center the color image on the canvas
        canvas_width = color_canvas.winfo_width()
        canvas_height = color_canvas.winfo_height()
        new_width, new_height = color_image.size
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        color_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=color_tk_image)
        color_canvas.image = color_tk_image

def handle_no_blobs():
    # Clear the blob and color canvases
    blob_canvas.delete("all")
    color_canvas.delete("all")
    avg_rgb_label.config(text="Avg RGB: N/A")
    largest_blob_size_label.config(text="Largest Blob Size: N/A")
    largest_blob_coords_label.config(text="Largest Blob Coordinates: N/A")
    blob_count_label.config(text="Blob Count: 0")
    print("No blobs detected. Adjust the mask parameters.")

def handle_too_many_blobs():
    # Clear the blob and color canvases
    blob_canvas.delete("all")
    color_canvas.delete("all")
    avg_rgb_label.config(text="Avg RGB: N/A")
    largest_blob_size_label.config(text="Largest Blob Size: N/A")
    largest_blob_coords_label.config(text="Largest Blob Coordinates: N/A")
    blob_count_label.config(text=f"Blob Count: >{MAX_BLOBS}")
    print(f"Too many blobs detected (>{MAX_BLOBS}). Adjust the mask parameters.")

def update_label(value, label):
    label.config(text=f"{float(value):.1f}")

def update_slider_and_label(slider, label):
    value = slider.get()
    label.config(text=f"{value:.0f}")

def save_report():
    if loaded_image is None:
        print("No image loaded to generate a report.")
        return
    
    # Extract the base name of the loaded image file and use it as the default report name
    image_filename = os.path.splitext(os.path.basename(loaded_image.filename))[0]
    default_filename = f"{image_filename}_analysis_report.pdf"
    
    # Ask the user for a save location with the default filename
    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialfile=default_filename  # Set the default filename to the image file name
    )
    
    if not save_path:
        return
    
    # Create a PDF document
    doc = SimpleDocTemplate(save_path, pagesize=letter)
    pdf_canvas = canvas.Canvas(save_path)
    pdf_canvas.setTitle(f"{image_filename} Analysis Report")  # Set the document title

    styles = getSampleStyleSheet()
    story = []
    
    # Add title with the image file name
    story.append(Paragraph(f"{image_filename} Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Capture the image views
    zoom_img = capture_canvas(image_zoom_canvas)
    mask_img = capture_canvas(mask_canvas)
    blob_img = capture_canvas(blob_canvas)
    color_img = capture_canvas(color_canvas)
    
    # Function to maintain aspect ratio
    def maintain_aspect_ratio(img, target_width):
        width, height = img.size
        aspect_ratio = height / width
        target_height = target_width * aspect_ratio
        return target_width, target_height
    
    # Add images to report, maintaining aspect ratio
    target_width = 3 * inch
    
    # Zoom View
    zoom_width, zoom_height = maintain_aspect_ratio(Image.open(zoom_img), target_width)
    story.append(Paragraph("Zoom View", styles['Heading2']))
    story.append(ReportLabImage(zoom_img, width=zoom_width, height=zoom_height))
    story.append(Spacer(1, 12))
    
    # Mask View
    mask_width, mask_height = maintain_aspect_ratio(Image.open(mask_img), target_width)
    story.append(Paragraph("Mask View", styles['Heading2']))
    story.append(ReportLabImage(mask_img, width=mask_width, height=mask_height))
    story.append(Spacer(1, 12))
    
    # Blob View
    blob_width, blob_height = maintain_aspect_ratio(Image.open(blob_img), target_width)
    story.append(Paragraph("Blob View", styles['Heading2']))
    story.append(ReportLabImage(blob_img, width=blob_width, height=blob_height))
    story.append(Spacer(1, 12))
    
    # Color View
    color_width, color_height = maintain_aspect_ratio(Image.open(color_img), target_width)
    story.append(Paragraph("Color View", styles['Heading2']))
    story.append(ReportLabImage(color_img, width=color_width, height=color_height))
    story.append(Spacer(1, 12))
    
    # Add slider parameters
    story.append(Paragraph("Slider Parameters", styles['Heading2']))
    slider_parameters = f"""
    Zoom: {zoom_slider.get()}<br/>
    Hue Lower: {hue_lower_slider.get()}<br/>
    Hue Upper: {hue_upper_slider.get()}<br/>
    Saturation Lower: {sat_lower_slider.get()}<br/>
    Saturation Upper: {sat_upper_slider.get()}<br/>
    Brightness Lower: {bri_lower_slider.get()}<br/>
    Brightness Upper: {bri_upper_slider.get()}<br/>
    """
    story.append(Paragraph(slider_parameters, styles['BodyText']))
    story.append(Spacer(1, 12))
    
    # Add dynamic variables
    story.append(Paragraph("Dynamic Variables", styles['Heading2']))
    dynamic_variables = f"""
    {avg_rgb_label.cget("text")}<br/>
    {largest_blob_coords_label.cget("text")}<br/>
    {largest_blob_size_label.cget("text")}<br/>
    {blob_count_label.cget("text")}<br/>
    """
    story.append(Paragraph(dynamic_variables, styles['BodyText']))
    
    # Build the document
    doc.build(story)
    print(f"Report saved to: {save_path}")

def capture_canvas(canvas):
    # Get the coordinates of the canvas relative to the screen
    canvas.update()  # Ensure the canvas is up-to-date
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    # Take a screenshot of the area covered by the canvas
    image = pyautogui.screenshot(region=(x, y, width, height))

    # Convert the screenshot to a format suitable for ReportLab
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)  # Rewind the buffer to the beginning

    return image_buffer

# Create the main window
root = tk.Tk()
root.title("Colorific analyzer v0.1")

# Initialize global variables
loaded_image = None
resized_image = None
current_mask = None
largest_blob_rgb = None
current_zoom_level = 10  # Default zoom level
visible_region = None

# Create a frame for parameter selection sliders
slider_frame = tk.Frame(root, width=200)  # Reduced width of the slider frame
slider_frame.grid(row=0, column=0, rowspan=2, sticky="nswe", padx=10, pady=10)

# Zoom Slider
zoom_label = tk.Label(slider_frame, text="Zoom")
zoom_label.pack(pady=(5, 0))

zoom_value_label = tk.Label(slider_frame, text="10.0")
zoom_value_label.pack(pady=(0, 5))

zoom_slider = ttk.Scale(slider_frame, from_=1, to=100, orient='horizontal',
                        command=lambda val: [update_zoom_image(float(val)), update_label(val, zoom_value_label)])
zoom_slider.set(10)  # Set default zoom level to 10 (20%)
zoom_slider.pack(pady=5)

# Hue Sliders
hue_label = tk.Label(slider_frame, text="Hue Lower/Upper")
hue_label.pack(pady=(5, 0))

hue_lower_value_label = tk.Label(slider_frame, text="0")
hue_lower_value_label.pack()

hue_lower_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(hue_lower_slider, hue_lower_value_label)])
hue_lower_slider.set(0)  # Default hue lower bound
hue_lower_slider.pack(pady=(0, 5))

hue_upper_value_label = tk.Label(slider_frame, text="255")
hue_upper_value_label.pack()

hue_upper_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(hue_upper_slider, hue_upper_value_label)])
hue_upper_slider.set(255)  # Default hue upper bound (set to max)
hue_upper_slider.pack(pady=(0, 5))

# Saturation Sliders
sat_label = tk.Label(slider_frame, text="Saturation Lower/Upper")
sat_label.pack(pady=(5, 0))

sat_lower_value_label = tk.Label(slider_frame, text="0")
sat_lower_value_label.pack()

sat_lower_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(sat_lower_slider, sat_lower_value_label)])
sat_lower_slider.set(0)  # Default saturation lower bound
sat_lower_slider.pack(pady=(0, 5))

sat_upper_value_label = tk.Label(slider_frame, text="255")
sat_upper_value_label.pack()

sat_upper_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(sat_upper_slider, sat_upper_value_label)])
sat_upper_slider.set(255)  # Default saturation upper bound (set to max)
sat_upper_slider.pack(pady=(0, 5))

# Brightness Sliders
bri_label = tk.Label(slider_frame, text="Brightness Lower/Upper")
bri_label.pack(pady=(5, 0))

bri_lower_value_label = tk.Label(slider_frame, text="200")
bri_lower_value_label.pack()

bri_lower_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(bri_lower_slider, bri_lower_value_label)])
bri_lower_slider.set(200)  # Default brightness lower bound
bri_lower_slider.pack(pady=(0, 5))

bri_upper_value_label = tk.Label(slider_frame, text="255")
bri_upper_value_label.pack()

bri_upper_slider = ttk.Scale(slider_frame, from_=0, to=255, orient='horizontal',
                             command=lambda val: [update_mask_view(), update_blob_view(), update_slider_and_label(bri_upper_slider, bri_upper_value_label)])
bri_upper_slider.set(255)  # Default brightness upper bound (set to max)
bri_upper_slider.pack(pady=(0, 5))

# Add Load and Save buttons
load_button = tk.Button(slider_frame, text="Load", command=load_action)
load_button.pack(side="left", pady=10, padx=5)
save_button = tk.Button(slider_frame, text="Save", command=save_report)
save_button.pack(side="right", pady=10, padx=5)

# Create frames for the 2x2 image view matrix
image_frame = tk.Frame(root)
image_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

# Add titles above the canvases
zoom_label_title = tk.Label(image_frame, text="Zoom")
zoom_label_title.grid(row=0, column=0, padx=5, pady=(0, 5))

mask_label_title = tk.Label(image_frame, text="Mask")
mask_label_title.grid(row=0, column=1, padx=5, pady=(0, 5))

blob_label_title = tk.Label(image_frame, text="Blob")
blob_label_title.grid(row=2, column=0, padx=5, pady=(0, 5))

color_label_title = tk.Label(image_frame, text="Color")
color_label_title.grid(row=2, column=1, padx=5, pady=(0, 5))

# Create canvas for image zoom view
image_zoom_canvas = tk.Canvas(image_frame, borderwidth=2, relief="groove", bg="white")
image_zoom_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# Create canvas for mask view
mask_canvas = tk.Canvas(image_frame, borderwidth=2, relief="groove", bg="white")
mask_canvas.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

# Create canvas for blob view
blob_canvas = tk.Canvas(image_frame, borderwidth=2, relief="groove", bg="white")
blob_canvas.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

# Create canvas for color view
color_canvas = tk.Canvas(image_frame, borderwidth=2, relief="groove", bg="white")
color_canvas.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")

# Add dynamic variable section
variables_frame = tk.Frame(root)
variables_frame.grid(row=2, column=0, columnspan=1, sticky="we", padx=10, pady=10)

avg_rgb_label = tk.Label(variables_frame, text="Avg RGB: N/A")
avg_rgb_label.pack(anchor="w")

largest_blob_coords_label = tk.Label(variables_frame, text="Largest Blob Coordinates: N/A")
largest_blob_coords_label.pack(anchor="w")

largest_blob_size_label = tk.Label(variables_frame, text="Largest Blob Size: N/A")
largest_blob_size_label.pack(anchor="w")

blob_count_label = tk.Label(variables_frame, text="Blob Count: 0")
blob_count_label.pack(anchor="w")

# Create info box for terminal messages with reduced height
info_box = tk.Text(root, height=8, width=50, wrap=tk.WORD)  # Reduced height of info box
info_box.grid(row=2, column=1, sticky="we", padx=10, pady=10)
info_box.insert(tk.END, "Info Box:\n")
info_box.config(state=tk.DISABLED)  # Initially disable editing

# Redirect standard output to the info box
info_box.config(state=tk.NORMAL)
sys.stdout = TextRedirector(info_box)
sys.stderr = TextRedirector(info_box)  # Also capture standard error

# Configure grid weights for responsive resizing
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
image_frame.grid_columnconfigure(0, weight=1)
image_frame.grid_columnconfigure(1, weight=1)
image_frame.grid_rowconfigure(1, weight=1)
image_frame.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(2, weight=1)

# Start the main loop
root.mainloop()
