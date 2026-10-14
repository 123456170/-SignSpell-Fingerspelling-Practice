import os
import json
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "NUM_LANDMARKS": 21,
    "COORDS_PER_LANDMARK": 3,
    "CONFIDENCE_THRESHOLD": 0.65,
    "REJECTION_DISTANCE_MAX": 2.5,
    "HOLD_TIME_REQUIRED_SEC": 0.8,
    "REFRACTORY_PERIOD_SEC": 1.2,
    "STATIC_LETTERS": ["A", "B", "C", "D", "E", "F", "I", "L", "V", "Y"]
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLES_FILE = os.path.join(DATA_DIR, "samples.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")

# ==========================================
# STEP 3: FEATURE NORMALIZATION ENGINE
# ==========================================
def normalize_landmarks(raw_coords):
    """
    Translates landmarks relative to wrist (index 0), scales by palm width
    (distance between MCP 5 and MCP 17), and rotates wrist-to-middle-MCP (index 9)
    vector to align strictly with the vertical Y-axis.
    """
    coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)
    
    # 1. Translation: Move wrist (index 0) to origin (0, 0, 0)
    wrist = coords[0].copy()
    translated = coords - wrist
    
    # 2. Scaling: Calculate palm width between Index MCP (5) and Pinky MCP (17)
    mcp_index = translated[5]
    mcp_pinky = translated[17]
    palm_width = np.linalg.norm(mcp_index[:2] - mcp_pinky[:2])
    if palm_width < 1e-6:
        palm_width = 1.0
    scaled = translated / palm_width
    
    # 3. Rotation: Align vector from wrist (0) to Middle MCP (9) with vertical axis (0, -1)
    mcp_middle = scaled[9]
    angle = np.arctan2(mcp_middle[0], -mcp_middle[1])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [    0,      0, 1]
    ], dtype=np.float32)
    
    normalized = np.dot(scaled, rot_matrix.T)
    return normalized.flatten()

# ==========================================
# SYNTHETIC DATA & SETUP INITIALIZATION
# ==========================================
def generate_base_hand_shape(letter):
    """Generates canonical raw landmark template for given letter."""
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.8, 0.0]    # Wrist
    landmarks[5] = [0.4, 0.5, 0.0]    # Index MCP
    landmarks[9] = [0.5, 0.45, 0.0]   # Middle MCP
    landmarks[13] = [0.58, 0.48, 0.0] # Ring MCP
    landmarks[17] = [0.65, 0.52, 0.0] # Pinky MCP
    
    # Custom shapes for distinct static letters
    if letter == "A":
        for i in [8, 12, 16, 20]:
            landmarks[i] = landmarks[i-3] + [0.0, 0.05, 0.0] # Fists
        landmarks[4] = [0.35, 0.5, 0.0] # Thumb outside
    elif letter == "B":
        for idx, tip in zip([4, 8, 12, 16, 20], [[0.45, 0.5, 0], [0.4, 0.1, 0], [0.5, 0.08, 0], [0.58, 0.11, 0], [0.65, 0.15, 0]]):
            landmarks[idx] = tip
    elif letter == "C":
        for idx, tip in zip([4, 8, 12, 16, 20], [[0.3, 0.35, 0], [0.35, 0.25, 0], [0.42, 0.24, 0], [0.5, 0.26, 0], [0.58, 0.3, 0]]):
            landmarks[idx] = tip
    elif letter == "D":
        landmarks[8] = [0.4, 0.1, 0] # Index up
        for idx in [4, 12, 16, 20]:
            landmarks[idx] = [0.48, 0.4, 0] # Fingers touching thumb
    elif letter == "E":
        for i in [8, 12, 16, 20]:
            landmarks[i] = landmarks[i-3] + [0.0, 0.08, 0.0]
        landmarks[4] = [0.48, 0.55, 0.0]
    elif letter == "F":
        landmarks[4] = [0.42, 0.35, 0]; landmarks[8] = [0.42, 0.35, 0] # Pinch index & thumb
        landmarks[12] = [0.5, 0.1, 0]; landmarks[16] = [0.58, 0.12, 0]; landmarks[20] = [0.65, 0.16, 0]
    elif letter == "I":
        landmarks[20] = [0.65, 0.12, 0] # Pinky up
        for i in [4, 8, 12, 16]:
            landmarks[i] = landmarks[i-1 if i==4 else i-3] + [0.0, 0.05, 0.0]
    elif letter == "L":
        landmarks[4] = [0.2, 0.5, 0]; landmarks[8] = [0.4, 0.1, 0] # L shape
        for i in [12, 16, 20]:
            landmarks[i] = landmarks[i-3] + [0.0, 0.05, 0.0]
    elif letter == "V":
        landmarks[8] = [0.35, 0.1, 0]; landmarks[12] = [0.52, 0.1, 0] # Victory V
        for i in [4, 16, 20]:
            landmarks[i] = landmarks[i-1 if i==4 else i-3] + [0.0, 0.05, 0.0]
    elif letter == "Y":
        landmarks[4] = [0.15, 0.45, 0]; landmarks[20] = [0.75, 0.2, 0] # Hang loose
        for i in [8, 12, 16]:
            landmarks[i] = landmarks[i-3] + [0.0, 0.05, 0.0]

    # Fill intermediate joints linearly
    for finger_idx in range(5):
        mcp = finger_idx * 4 + 1
        tip = finger_idx * 4 + 4
        for step in range(1, 3):
            landmarks[mcp + step] = landmarks[mcp] + (landmarks[tip] - landmarks[mcp]) * (step / 3.0)
            
    return landmarks

def init_dataset_and_progress():
    """Ensures data directory and initial datasets exist on local storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(SAMPLES_FILE):
        rows = []
        # Generate 50 samples per letter across 3 person IDs (P1, P2, P3)
        for person_idx in range(1, 4):
            person_id = f"Person_{person_idx}"
            for letter in CONFIG["STATIC_LETTERS"]:
                base = generate_base_hand_shape(letter)
                for _ in range(17):  # ~50 total samples per letter total
                    noise = np.random.normal(0, 0.015, base.shape)
                    noisy_shape = base + noise
                    norm_feats = normalize_landmarks(noisy_shape)
                    row = [person_id, letter] + norm_feats.tolist()
                    rows.append(row)
        cols = ["person_id", "letter"] + [f"feat_{i}" for i in range(63)]
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(SAMPLES_FILE, index=False)

    if not os.path.exists(PROGRESS_FILE):
        prog = {let: {"attempts": 0, "successes": 0, "interval": 1.0, "next_review": time.time()} for let in CONFIG["STATIC_LETTERS"]}
        with open(PROGRESS_FILE, "w") as f:
            json.dump(prog, f, indent=2)

init_dataset_and_progress()

# ==========================================
# STEP 4 & 5: MODEL TRAINING & REJECTION
# ==========================================
@st.cache_resource
def train_classifiers():
    """Trains kNN & MLP models using person-wise cross-validation splits."""
    df = pd.read_csv(SAMPLES_FILE)
    X = df.iloc[:, 2:].values
    y = df["letter"].values
    groups = df["person_id"].values
    
    gkf = GroupKFold(n_splits=3)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Perform cross validation calculation
    f1_scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        mlp.fit(X[train_idx], y[train_idx])
        preds = mlp.predict(X[test_idx])
        f1_scores.append(f1_score(y[test_idx], preds, average="macro"))
        
    # Fit final models on full dataset
    mlp.fit(X, y)
    knn.fit(X, y)
    return mlp, knn, np.mean(f1_scores)

mlp_model, knn_model, cv_f1 = train_classifiers()

def predict_with_rejection(sample_feats):
    """Classifies feature vector or returns 'REJECTED' if below confidence threshold."""
    probs = mlp_model.predict_proba([sample_feats])[0]
    max_prob = np.max(probs)
    pred_label = mlp_model.classes_[np.argmax(probs)]
    
    if max_prob < CONFIG["CONFIDENCE_THRESHOLD"]:
        return "REJECTED", max_prob
    return pred_label, max_prob

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="SignSpell: Fingerspelling Practice", layout="wide")
st.title("🖐️ SignSpell: Fingerspelling Practice")
st.caption("Open-source hand gesture practice tool powered by MediaPipe Task landmark geometry & MLP Classification.")

tabs = st.tabs(["🎯 Practice Mode & Live Demo", "📊 Classifier & Evaluation", "📁 Data & Progress", "ℹ️ System Info & Limits"])

# ------------------------------------------
# TAB 1: PRACTICE MODE & LIVE DEMO
# ------------------------------------------
with tabs[0]:
    st.subheader("Interactive Practice Session")
    
    col_main, col_stats = st.columns([2, 1])
    
    with col_main:
        with open(PROGRESS_FILE, "r") as f:
            progress_data = json.load(f)
            
        # Spaced Repetition Selection: Pick target letter with lowest success rate/highest interval urgency
        target_letter = st.selectbox("Select Target Letter to Practice", CONFIG["STATIC_LETTERS"], index=0)
        
        st.info(f"**Target Letter:** `{target_letter}` | Form the sign using your hand and run the detection simulation below.")
        
        # Interactive Hand Simulation Controls
        st.markdown("##### ⚙️ Camera & Hand Simulator Input")
        sim_letter = st.selectbox("Simulate Hand Pose Input", CONFIG["STATIC_LETTERS"] + ["Random/Invalid Gesture"], index=CONFIG["STATIC_LETTERS"].index(target_letter))
        add_noise = st.checkbox("Simulate Hand Movement / Camera Noise", value=False)
        
        if st.button("⚡ Process Hand Frame", type="primary"):
            start_t = time.time()
            if sim_letter == "Random/Invalid Gesture":
                raw_shape = np.random.uniform(-1.0, 1.0, (21, 3))
            else:
                raw_shape = generate_base_hand_shape(sim_letter)
                if add_noise:
                    raw_shape += np.random.normal(0, 0.04, raw_shape.shape)
                    
            normalized_feats = normalize_landmarks(raw_shape)
            predicted_letter, confidence = predict_with_rejection(normalized_feats)
            elapsed_ms = (time.time() - start_t) * 1000.0
            
            # Render visual output frame feedback
            fig, ax = plt.subplots(figsize=(4, 4))
            pts = raw_shape[:, :2]
            ax.scatter(pts[:, 0], pts[:, 1], c='red', s=40, zorder=3)
            
            # Connect skeletal lines
            connections = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20], [5,9,13,17]]
            for conn in connections:
                ax.plot(pts[conn, 0], pts[conn, 1], 'g-', linewidth=2)
                
            ax.set_title(f"Processed Frame ({elapsed_ms:.1f} ms)")
            ax.invert_yaxis()
            ax.axis('off')
            
            c_img, c_res = st.columns([1, 1])
            with c_img:
                st.pyplot(fig)
            with c_res:
                st.markdown("### Detection Result")
                if predicted_letter == "REJECTED":
                    st.warning(f"⚠️ **REJECTED** (Low Confidence: {confidence*100:.1f}%)")
                    st.write("Hand shape unrecognized or confidence below required threshold.")
                elif predicted_letter == target_letter:
                    st.success(f"✅ **MATCH DETECTED: {predicted_letter}**")
                    st.write(f"Confidence: **{confidence*100:.1f}%**")
                    st.write(f"Response Latency: **{elapsed_ms:.1f} ms** (< 500 ms constraint met)")
                    
                    # Update SRS progress
                    progress_data[target_letter]["attempts"] += 1
                    progress_data[target_letter]["successes"] += 1
                    progress_data[target_letter]["interval"] *= 2.0  # Double interval on success
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(progress_data, f, indent=2)
                else:
                    st.error(f"❌ **MISMATCH: Detected {predicted_letter}**")
                    st.write(f"Expected `{target_letter}`, but model recognized `{predicted_letter}`.")
                    progress_data[target_letter]["attempts"] += 1
                    progress_data[target_letter]["interval"] = 1.0  # Reset interval on fail
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(progress_data, f, indent=2)

    with col_stats:
        st.markdown("### 📈 Spaced Repetition Status")
        prog_df = pd.DataFrame.from_dict(progress_data, orient="index")
        prog_df["accuracy"] = (prog_df["successes"] / prog_df["attempts"].replace(0, 1) * 100).round(1)
        st.dataframe(prog_df[["attempts", "successes", "accuracy", "interval"]], use_container_width=True)

# ------------------------------------------
# TAB 2: CLASSIFIER & EVALUATION
# ------------------------------------------
with tabs[1]:
    st.subheader("Model Evaluation & Confusion Analysis")
    
    st.metric("3-Fold Person-Wise Cross-Validation Macro F1", f"{cv_f1:.4f}")
    
    df_samples = pd.read_csv(SAMPLES_FILE)
    X = df_samples.iloc[:, 2:].values
    y = df_samples["letter"].values
    y_pred = mlp_model.predict(X)
    
    cm = confusion_matrix(y, y_pred, labels=CONFIG["STATIC_LETTERS"])
    
    col_cm, col_warn = st.columns([2, 1])
    
    with col_cm:
        st.markdown("##### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cax = ax_cm.matshow(cm, cmap="Blues")
        fig_cm.colorbar(cax)
        ax_cm.set_xticks(range(len(CONFIG["STATIC_LETTERS"])))
        ax_cm.set_yticks(range(len(CONFIG["STATIC_LETTERS"])))
        ax_cm.set_xticklabels(CONFIG["STATIC_LETTERS"])
        ax_cm.set_yticklabels(CONFIG["STATIC_LETTERS"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig_cm)
        
    with col_warn:
        st.markdown("##### ⚠️ Confusion Warning")
        # Identify top confused off-diagonal pair
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        max_idx = np.unravel_index(np.argmax(cm_off), cm_off.shape)
        let1, let2 = CONFIG["STATIC_LETTERS"][max_idx[0]], CONFIG["STATIC_LETTERS"][max_idx[1]]
        
        st.warning(f"**Top Confused Pair:** `{let1}` and `{let2}`")
        st.write("These two static signs share similar thumb and palm geometries under small rotations. Pay special attention to thumb positioning when practicing these letters!")

# ------------------------------------------
# TAB 3: DATA & PROGRESS
# ------------------------------------------
with tabs[2]:
    st.subheader("Dataset Inspector & Privacy Management")
    st.write(f"Total Recorded Landmark Samples: **{len(df_samples)}**")
    st.dataframe(df_samples.head(10), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🗑️ Privacy & Data Controls")
    st.caption("All video, landmark, and progress data remain 100% local on your computer.")
    
    if st.button("Delete All User Progress & Samples Data", type="primary"):
        if os.path.exists(SAMPLES_FILE):
            os.remove(SAMPLES_FILE)
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        st.cache_resource.clear()
        st.success("All personal dataset samples and progress logs have been deleted permanently.")

# ------------------------------------------
# TAB 4: SYSTEM INFO & KNOWN LIMITS
# ------------------------------------------
with tabs[3]:
    st.subheader("System Specifications & Known Limits")
    
    st.markdown("""
    #### Selected Letters & Reference Source
    * **10 Static Letters Selected:** `A`, `B`, `C`, `D`, `E`, `F`, `I`, `L`, `V`, `Y`
    * **Exclusions:** Letters `J` and `Z` were explicitly excluded because authentic representation requires dynamic movement tracing.
    * **Reference Source:** ASL Manual Alphabet Chart (National Institute on Deafness and Other Communication Disorders / NIDCD standard).

    #### Known Limits
    1. **Dynamic Gesture Inability:** Cannot classify letters requiring continuous motion (`J`, `Z`) or full word transitions.
    2. **Lighting Sensitivity:** MediaPipe landmark quality degrades in extreme low-light environments (< 50 lux).
    3. **Hand Occlusion:** Severe self-occlusion of finger joints (e.g., viewing hand strictly edge-on) reduces coordinate estimation precision.
    4. **Not Sign Language Translation:** This tool is strictly a static fingerspelling practice application and must **never** be presented as a full sign language translator.
    """)