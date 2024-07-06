# Real-Time-Student-Grading-System-with-Transfer-Learning-
Real-Time Student Grading System Project with Transfer Learning 


## Installation

To run this project, you need to have Python installed. Follow the steps below to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/txitxy/SeminProject.git
    cd SeminProject
    ```

2. Install the required libraries:
    ```bash
    pip install tensorflow flask opencv-python numpy pandas matplotlib
    ```

3. Train the model (if not using the provided `model.h5`):
    ```bash
    python train_model.py
    ```

## Running the Application

To start the Flask application, run the following command:
```bash
python app.py


Then, open your web browser and go to http://127.0.0.1:5000.

Project Features
Home Page
Description of the project.
Navigation links to other features.
Student Profile Page
Displays details of the student profile.
Real-Time Detection Page
Uses OpenCV to capture and process video in real-time.
Applies the trained model to identify and label the student's face.
Folder Structure
app.py: Main application file for Flask.
templates/: HTML templates for the web pages.
index.html: Home page.
profile.html: Student profile page.
detect.html: Real-time detection page.
static/: Static files such as CSS and images.
styles.css: CSS file for styling.
styleImages/: Images used for backgrounds and styling.
bckgrnd.jpg: Background image for the home and detection pages.
bckgrnd2.jpg: Background image for the profile page.
model/: Contains the trained model file model.h5.
data/: Directory for training and validation datasets.
train/: Training dataset.
student/: Images of students.
other/: Images of other categories.
validation/: Validation dataset.
student/: Images of students.
other/: Images of other categories.
train_model.py: Script to train the model.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Special thanks to Professor Stiven Fortes for the guidance and support during the development of this project.




