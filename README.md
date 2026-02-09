# Automated Human Profiling System 
### Image-Based Attribute Estimation using Multi-Head ResNet50

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automated-profile-generation.streamlit.app/)

An end-to-end Computer Vision system designed to automate human profiling. This project utilizes Deep Learning to simultaneously estimate five key human attributes **Age, Gender, Ethnicity, Height, and Weight** from a single digital image.

## Key Features
* **Multi-Task Learning (MTL):** A single Multi-Head ResNet50 architecture handles 3 classification tasks and 2 regression tasks simultaneously.
* **Localized Demographic Accuracy:** Specifically optimized for the Malaysian demographic (Southeast Asian, East Asian, Indian, and White).
* **Face Detection Gatekeeper:** Integrated OpenCV Haar Cascade to ensure only valid human subjects are processed.
* **Real-time Inference:** Optimized pipeline providing full results in under 2 seconds.
* **Interactive Web UI:** Built with Streamlit for seamless user interaction and deployment.

## System Preview
![System Demo](link_to_your_screenshot.png)
*Example output showing the automated generation of a human profile with prediction speed metrics.*

## Technical Stack
* **Language:** Python 3.10+
* **AI Framework:** PyTorch (Torchvision)
* **Computer Vision:** OpenCV
* **Frontend:** Streamlit
* **Preprocessing:** Pandas, NumPy, PIL

## Model Architecture
The system uses a **ResNet50 backbone** as a shared feature extractor. The final layer is branched into five specialized heads:
1.  **Age Head:** 5-class classification.
2.  **Gender Head:** Binary classification.
3.  **Ethnicity Head:** 4-class localized classification.
4.  **Height Head:** Linear regression (cm).
5.  **Weight Head:** Linear regression (kg).

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   Install dependencies:
2. Install dependencies:
pip install -r requirements.txt
3. Run the application:
streamlit run profile_app.py

Dataset Attribution:
FairFace: For demographic classification (Age, Gender, Ethnicity).
Celeb-FBI: For physical attribute regression (Height, Weight).

Developed as a Final Year Project at the Faculty of Computer Science and Information Technology, UPM.
