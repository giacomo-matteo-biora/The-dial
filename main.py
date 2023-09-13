import os
from tkinter import *
from tkinter import filedialog
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import webbrowser as wb

# Function to open GitHub credits link
def open_github_credits():
    wb.open('https://github.com/giacomo-matteo-biora')

# Function to open Hugging Face documentation link
def open_huggingface():
    wb.open('https://huggingface.co/google/vit-base-patch16-224')

# Function to select the input directory
def browseInputDirectory():
    global InputDirectory
    InputDirectory = filedialog.askdirectory(initialdir="/", title="Select a Directory")

# Function to select the output directory
def browseOutputDirectory():
    global OutputDirectory
    OutputDirectory = filedialog.askdirectory(initialdir="/", title="Select a Directory")

# Function to perform image classification and organize images
def test():
    threshold_value = float(threshold_entry.get())
    label_photo_text = label_photo_entry.get()

    # Input directory containing images
    input_directory = InputDirectory

    # Output directory for mountain bike images
    output_directory = OutputDirectory

    # Load the model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process all images in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_directory, filename)
            image = Image.open(image_path)

            # Perform inference with the model
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

            # Get the predicted score
            predicted_score = logits.softmax(dim=-1)[0, predicted_class_idx].item()

            # Get the predicted label
            predicted_class_label = model.config.id2label[predicted_class_idx]

            # Check if the predicted score exceeds the threshold
            if predicted_score >= threshold_value and label_photo_text in predicted_class_label.lower():
                # Move the image to the mountain bike output directory
                output_path = os.path.join(output_directory, filename)
                os.rename(image_path, output_path)
                result_text.insert(INSERT, f"{filename} is a mountain bike with a score of {predicted_score:.2f} and has been moved to the output directory (Label: {predicted_class_label}).\n")
            else:
                result_text.insert(INSERT, f"{filename} is not a mountain bike with a score of {predicted_score:.2f} (Label: {predicted_class_label}).\n")

# Create the main window
root = Tk()
root.geometry('400x500')
root.title('PHOTO ORGANIZATION AI')

# Add a menu
menu = Menu(root)
root.config(menu=menu)

# Create "Credits" submenu
helpmenu = Menu(menu)
menu.add_cascade(label='Credits', menu=helpmenu)
helpmenu.add_command(label='By Giacomo Matteo Biora', command=open_github_credits)

# Create "Help" submenu
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='Label Problem', command=open_huggingface)

# Create widgets
button = Button(root, bg='green', fg='black', text='START!', font=("Times New Roman", 30), relief=FLAT, borderwidth=0,
                command=test)
button.configure(width=20, height=2, activebackground="green")
button.pack(side='bottom', pady=10)

button_explore_input_directory = Button(root, bg='white', fg='black', text='Input Directory',
                                        command=browseInputDirectory)
button_explore_input_directory.configure(width=40, height=4, activebackground="green")
button_explore_input_directory.pack(side='top', padx=10, pady=5)

button_explore_output_directory = Button(root, bg='white', fg='black', text='Output Directory',
                                         command=browseOutputDirectory)
button_explore_output_directory.configure(width=40, height=4, activebackground="green")
button_explore_output_directory.pack(side='top', padx=10, pady=5)

threshold_label = Label(root, text="Threshold Value From 0.1 to 1.0")
threshold_label.pack(side="top", padx=10, pady=10)
threshold_entry = Entry(root)
threshold_entry.pack(side="top", padx=10, pady=10)

label_photo_label = Label(root, text="Set Label")
label_photo_label.pack(side="top", padx=10, pady=10)
label_photo_entry = Entry(root)
label_photo_entry.pack(side="top", padx=10, pady=10)

result_text = Text(root, wrap=WORD)
result_text.pack(side="top", padx=10, pady=10)

# Run the main window
root.mainloop()
