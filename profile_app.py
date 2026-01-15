import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import cv2

# This is a safer way to load the XML on Windows
import os
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    st.error("OpenCV data files not found. Please check your opencv-python installation.")
else:
    face_cascade = cv2.CascadeClassifier(cascade_path)

# --- 1. MODEL ARCHITECTURES (From your Training Code) ---
class MultiHeadResNet(nn.Module):
    def __init__(self, num_age=5, num_gender=2, num_eth=4):
        super().__init__()
        self.base = models.resnet50(weights=None)
        feat_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.age_head = nn.Linear(feat_dim, num_age)
        self.gender_head = nn.Linear(feat_dim, num_gender)
        self.ethnicity_head = nn.Linear(feat_dim, num_eth)

    def forward(self, x):
        x = self.base(x)
        return self.age_head(x), self.gender_head(x), self.ethnicity_head(x)


class BodyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=None)
        feat_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.h_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1))
        self.w_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, x):
        feat = self.base(x)
        return self.h_head(feat), self.w_head(feat)


# --- 2. CONFIGURATION & PRE-PROCESSING ---
H_MEAN = 171.07625482625483
H_STD = 11.69763785255054
W_MEAN = 65.67535392535393
W_STD = 15.924241725520226

AGE_CLASSES = ["Child (0-9)", "Youth (10-19)", "Adult (20-39)", "Middle-Aged (40-59)", "Senior (60+)"]
GENDER_CLASSES = ["Male", "Female"]
ETHNICITY_CLASSES = ["East Asian", "Southeast Asian", "Indian", "White"]


def pad_to_square(image):
    w, h = image.size
    max_wh = max(w, h)
    result = Image.new("RGB", (max_wh, max_wh), (0, 0, 0))
    result.paste(image, ((max_wh - w) // 2, (max_wh - h) // 2))
    return result


face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

body_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. PAGE DESIGN (Dark Theme & White Card) ---
st.set_page_config(page_title="Automated Profile Generation", layout="wide")

st.markdown("""
    <style>
    /* 1. Overall Dark Page Background */
    .stApp { background-color: #1a1a1a; }

    /* 2. The White Card Container */
    [data-testid="column"] {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        color: #000000 !important;
    }

    /* 3. FIXING READABILITY: Targeting Labels specifically */
    /* This makes 'HEIGHT', 'WEIGHT', etc. visible */
    .stWidgetLabel p {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
    }

    /* 4. The Preview text color */
    .preview-text {
        color: white !important;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* 5. Input Box Styling */
    .stTextInput input {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        border: 2px solid #333 !important;
    }

    /* 6. Title Styling */
    h1 { color: #ffffff !important; text-align: center; margin-bottom: 30px; }

    /* 7. Button Styling */
    .stButton>button {
        background-color: #333 !important;
        color: white !important;
        width: 100%;
        border-radius: 0px;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_all_models():
    f_model = MultiHeadResNet()
    f_model.load_state_dict(torch.load('Best_Face_Resnet50model.pth', map_location='cpu'))
    f_model.eval()

    b_model = BodyResNet()
    b_model.load_state_dict(torch.load('Best_Body_Resnet50model.pth', map_location='cpu'))
    b_model.eval()
    return f_model, b_model


model_face, model_body = load_all_models()

#Sidebar guide
with st.sidebar:
    st.title("User Guide")
    st.info("""
    1. **Upload** a high-resolution image (.jpg, .jpeg, .png).
    2. Ensure the **face is visible** and well-lit.
    3. Click **'GENERATE PROFILE'** to run AI prediction.
    4. Use **'TRY ANOTHER'** to clear memory.
    """)
    st.warning("⚠️ Note: System requires a clear face to generate profile metadata.")

# --- 5. UI LAYOUT ---
st.title("Automated Profile Generation")

col_img, col_form = st.columns([1, 1.5], gap="large")

with col_img:
    st.markdown("<p class='preview-text'>IMAGE PREVIEW</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width='stretch')
    else:
        st.image("https://via.placeholder.com/400x400.png?text=Preview", width='stretch')

with col_form:
    st.text_input("ID NUMBER:", value="A-001")

    # The value is pulled from session_state. If empty, it shows ""
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.text_input("AGE:", value=st.session_state.get('age', ''))
    with r1c2:
        st.text_input("GENDER:", value=st.session_state.get('gender', ''))

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.text_input("HEIGHT:", value=st.session_state.get('height', ''))
    with r2c2:
        st.text_input("ETHNICITY:", value=st.session_state.get('eth', ''))

    st.text_input("WEIGHT:", value=st.session_state.get('weight', ''))

    st.write("---")

    # Only show the download button if a prediction has actually been made
    if st.session_state.get('age'):
        #st.write("")  # Tiny gap

        # We create 2 columns so the button only takes up 50% width (same as the buttons above)
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            report_text = f"""
            USER PROFILE REPORT
            -------------------
            ID: A-001
            Age: {st.session_state.get('age', 'N/A')}
            Gender: {st.session_state.get('gender', 'N/A')}
            Ethnicity: {st.session_state.get('eth', 'N/A')}
            Height: {st.session_state.get('height', 'N/A')}
            Weight: {st.session_state.get('weight', 'N/A')}
            -------------------
            Generated by: Automated Profile System
            """

        st.download_button(
            label="DOWNLOAD PROFILE DATA",
            data=report_text,
            file_name=f"profile_result.txt",
            mime="text/plain"
        )

    # BUTTONS SECTION
    b_col1, b_col2 = st.columns(2)

    # Button 1: Predict
    if b_col1.button("GENERATE PROFILE"):
        if uploaded_file:
            start_time = time.time()
            with st.spinner('Analyzing...'):
                # 1. Prepare Images (NEW)
                pil_img = Image.open(uploaded_file).convert('RGB')
                opencv_img = np.array(pil_img)
                opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

                # 2. DETECT FACE
                faces = face_cascade.detectMultiScale(gray, 1.1, 8) #minNeighbors changed from 4

                if len(faces) == 0:
                    # 1. Show the warning
                    st.error("⚠️ No human face detected. Profile generation aborted.")

                    # 2. Clear old data from boxes so it doesn't look like the AI is guessing
                    for k in ['age', 'gender', 'eth', 'height', 'weight']:
                        st.session_state[k] = ""

                    # 3. Stop everything right here
                    st.stop()

                    # --- IF WE REACH HERE, IT MEANS A FACE WAS FOUND ---
                (x, y, w, h) = faces[0]
                pad = int(w * 0.2)
                crop_face = pil_img.crop((max(0, x - pad), max(0, y - pad),
                                          min(pil_img.width, x + w + pad),
                                          min(pil_img.height, y + h + pad)))

                # 3. PREDICTIONS
                # Face Tasks
                f_input = face_transform(crop_face).unsqueeze(0)
                with torch.no_grad():
                    a_out, g_out, e_out = model_face(f_input)
                    g_probs = torch.softmax(g_out, dim=1)

                    # Store results in session_state
                    if g_probs[0][1] > 0.40:
                        st.session_state['gender'] = "Female"
                    else:
                        st.session_state['gender'] = "Male"

                    st.session_state['age'] = AGE_CLASSES[torch.argmax(a_out).item()]
                    st.session_state['eth'] = ETHNICITY_CLASSES[torch.argmax(e_out).item()]

                # Body Tasks
                b_input = body_transform(pil_img).unsqueeze(0)
                with torch.no_grad():
                    ph, pw = model_body(b_input)
                    st.session_state['height'] = f"{round((ph.item() * H_STD) + H_MEAN)} cm"
                    st.session_state['weight'] = f"{round((pw.item() * W_STD) + W_MEAN)} kg"

                # Put this at the end of your prediction logic, before the rerun
                with st.sidebar:
                    st.markdown("### 🔍 AI RAW OUTPUT")
                    st.write(st.session_state)

                # Calculate total time
                duration = time.time() - start_time
                st.session_state['speed'] = f"{round(time.time() - start_time, 2)} seconds"

                st.rerun()

        else:
            st.error("Please upload an image!")

    # Button 2: Reset
    if b_col2.button("TRY ANOTHER IMAGE"):
        for k in ['age', 'gender', 'eth', 'height', 'weight']:
            if k in st.session_state:
                st.session_state[k] = ''  # Clear the values
        st.rerun()

# --- PUT THE SPEED CODE HERE (Inside col_form, at the bottom) ---
    if 'speed' in st.session_state:
        st.markdown(f"""
            <div style="text-align: left; color: #888; font-size: 0.8rem; margin-top: 10px;">
                Profile Generated in {st.session_state['speed']}
            </div>
        """, unsafe_allow_html=True)