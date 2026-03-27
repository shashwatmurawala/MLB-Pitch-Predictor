# ⚾️ MLB Pitch Predictor: Dual Brain Architecture

A cutting-edge machine learning web application built with React and Flask that predicts the exact pitch type and strike-zone location for any specific batter-pitcher matchup in Major League Baseball.

## 🚀 Key Features

*   **Dual-Brain Matchup Architecture:** Natively uses an ensemble of over **1,100 specialized Random Forest models** to independently evaluate both the Pitcher's specific tendencies and the Batter's specific weaknesses.
*   **Deep Sequence Memory:** Understands the exact flow of the at-bat, tracking the previous two pitches thrown and their exact zones to accurately anticipate pitching setup strategies.
*   **Dynamic Repertoire Masking:** Synthesizes the exact repertoire of any pitcher on the mound, strictly preventing predictions of impossible pitches (e.g., it will organically zero-out a Split-Finger prediction for a pitcher who doesn't possess one).
*   **Live Interactive UI:** A beautiful, responsive React web app visualizing real-time strike zones, live state-machine game states, and probabilistic pitch outcomes.

---

## 📊 Architecture & Data Process

The core machine learning brains generate predictions using robust tabular data incorporating over 15 game-state variables simultaneously (Balls, Strikes, Outs, Base Runners, Inning, Home/Away Score, Stand/Throws, etc.).

*   **Training Dataset:** Built using exhaustive play-by-play data encompassing literally **every single pitch thrown during the 2025 MLB Regular Season** (over 700,000 recorded pitches).
*   **Testing Dataset:** To ensure absolute validity and prevent data leakage, the finalized models were validated strictly against the entirely unseen **2025 MLB Postseason** dataset.

### 🏆 Testing Results (2025 MLB Postseason)

Because MLB pitching incorporates immense randomness, accurately guessing the identical pitch type *and* spatial location out of 129 possible combinations is incredibly difficult. Implementing our **Matchup-Based Dual Brain** (blending the Pitcher's brain against the Batter's brain with a 60/40 weighted split) generated massive accuracy boosts across the board:

1. **Exact Pitch + Strike Zone Location (129 Combinations)**
   *   **Top 1 Guess:** **12.66%** Exact Match Accuracy *(Massive jump from early 5% baseline)*
   *   **Top 5 Guesses:** **35.53%** Accuracy

2. **Pitch Category Anticiption (Fastball vs. Breaking vs. Offspeed)**
   *   Running the specialized `Grouped Dual Brain` over the Postseason achieved a staggering **55.31%** accuracy for anticipating the category of the arriving pitch out of the Pitcher's hand.

---

## 💻 Installation & Setup

### 1. Backend API (Flask)
Start the ML engine to compile and load the 1,100+ Python models.
```bash
git clone git@github.com:shashwatmurawala/MLB-Pitch-Predictor.git
cd MLB-Pitch-Predictor
source venv/bin/activate
pip install -r requirements.txt

python api.py
```
*The API will boot locally on `http://127.0.0.1:5001`. Do not shut this terminal down.*

### 2. Frontend Application (React/Vite)
Open a new terminal to start the visual dashboard and Live Game State tracker.
```bash
cd MLB-Pitch-Predictor/frontend
npm install
npm run dev
```
*The web app will cleanly launch locally on `http://localhost:5173`.*

---

## 🕹️ How to Use

1. **Fire it up:** Access the React application visually via your local browser.
2. **Select the Matchup:** Type in a specific configuration (e.g., *Gerrit Cole vs. Aaron Judge*) and the autocomplete directories will load them.
3. **Configure the Environment:** Recreate the exact live game variables (Runners on 1B/2B, 3-2 Full Count, 2 Outs).
4. **Program the Sequence:** Enter the pitch thrown prior (e.g. *4-Seam Fastball thrown High-Inside*).
5. **Simulate:** Click the **Predict Next Pitch** beacon!
6. The Flask Backend will instantly compute Gerrit Cole's specialized model, concurrently weigh it against Aaron Judge's specialized model, dynamically render the top outcomes to percentages, and paint the exact predicted target on your interactive Strike Zone!
