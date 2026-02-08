# ⚽ Football Analytics System – Baseline Project

## 📌 Project Description
This project is an **initial implementation (baseline)** of a football match analysis system using **computer vision and image-based analysis**.

The system processes a full football match video and automatically generates:
- A **processed match video** with player & ball tracking
- **Team-wise heatmaps** showing player movement intensity
- A **tactical minimap video** representing player & ball positions
- **Ball possession statistics** updated dynamically during the match

⚠️ This version focuses on **image analysis and rule-based logic**.  
Machine Learning integration is **planned for future phases**.

---

## 🧩 Project Phase – Current (Baseline Phase)

### Phase 1: Match Analysis (Baseline)
The current phase performs **end-to-end analysis of a football match video**, including:

### ✅ Features Implemented
- Player detection and tracking (YOLO-based)
- Team-wise player classification (jersey color analysis)
- Ball detection and temporal smoothing
- Ball possession estimation
- Tactical **minimap video generation**
- Team-wise **heatmap images**
- Referee & crowd filtering
- Goalkeeper handling (image-based logic)

### 📂 Outputs Generated
- `processed_<sample1>.mp4` → annotated match video  
- `minimap_<sample1>.mp4` → tactical minimap video  
- `heatmaps/heatmap_team_A.png`  
- `heatmaps/heatmap_team_B.png`
- **ALL THE OUTPUTS are available at**  : https://drive.google.com/drive/folders/1tbDYQBkMBFgrH0Txikf3CCoKXAT4uiLv

This phase serves as a **stable baseline** and reference point for future expansion.

---

## 🔮 Future Phases (Planned)

### Phase 2: Targeted Match Analysis  
**Example:**  
Manchester United & Real Madrid match analysis
- This will be an addition to get a analysis video of any match being played against this 2 teams.  <ins>**STARTING PHASE OF INTEGRATING ML**</ins>
- Planned improvements:
- Automated jersey color learning from reference images
- Advanced goalkeeper identification
- Improved ball interaction logic (passes, touches)
- Team shape & formation analysis
- Player role identification (defender, midfielder, forward)
- Enhanced possession logic using temporal context

---

### Phase 3: Universal Football Match Analysis
This phase aims to **lock all features from the start** and support:
- Any football match
- Any teams
- Any jersey colors
- Minimal manual configuration

Planned additions:
- Machine Learning–based team classification
- Player re-identification across camera cuts
- Event detection (passes, shots, interceptions)
- Match statistics summary (JSON / CSV)
- Tactical insights for analysts

---

## 🤖 Machine Learning Integration (Future Scope)
ML will be used to:
- Learn team jerseys automatically
- Improve ball detection under occlusion
- Stabilize player identities across frames
- Classify events (pass, shot, duel)
- Enhance possession accuracy

The current system is intentionally **kept image-based** to ensure stability and interpretability before ML integration.

---

## 🚀 How to Run the Project (Step-by-Step)

### 1️⃣ Clone the Repository
git clone <repository-url>
- cd 1.Football_Analytics
### 2️⃣ Create Virtual Environment (Recommended)
- python -m venv venv
Activate:
Windows
- venv\Scripts\activate
Linux / Mac
- source venv/bin/activate
### 3️⃣ Install Dependencies
pip install -r requirements.txt
### 4️⃣ Add Input Video
Place your match video inside:
- data/videos/samples/
- Example:
- sample1.mp4
- sample2.mp4
### 5️⃣ Run the Pipeline
- python -m run_pipeline
### 6️⃣ Check Outputs
All outputs will be saved inside:
data/output/
