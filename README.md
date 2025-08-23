# Machine-Learning-Based-Prediction-of-Combinational-Logic-Depth-Across-Technology-Nodes
Machine learning-based prediction of propagation delay and combinational logic depth across semiconductor nodes (14nm–180nm). The propagation delay model works robustly across all nodes, while the logic depth classifier is optimized for advanced nodes (14nm, 28nm) and is being refined for broader applicability. This project leverages data-driven models to estimate circuit depth, supporting faster and more efficient VLSI design optimization.

## INTRODUCTION
Every digital device you use—your phone, laptop, or smartwatch—runs on **logic circuits** built from simple gates like AND, OR, and NOT.  
One key property of these circuits is **logic depth**, which tells us **how many layers of gates a signal must pass through** before producing an output.  

- A **small depth** → faster circuits, lower power consumption, better efficiency.  
- A **large depth** → slower circuits, higher energy use, possible timing failures.  

As semiconductor technology scales down (from **180nm → 14nm chips**), circuits become more complex and harder to analyze.  
Traditionally, engineers estimate logic depth and timing through **manual simulations**, which are slow and resource-intensive.  

## PROJECT OBJECTIVE  
Develop ML models that:  
  - Predict **logic depth** across multiple technology nodes.  
  - Estimate **propagation delays** (timing analysis).  
  - Classify **slack violations** (timing reliability check).

This makes the design process **faster, smarter, and more automated**, helping engineers build efficient chips and paving the way for smarter applications—even in fields like **biomedical device circuits**.  

## Code 1 – Logic Depth Prediction
**Purpose:**
- This code uses machine learning regression models (Random Forest and Gradient Boosting) to estimate the logic depth of a digital circuit.
- Logic depth means how many layers of logic gates an input signal must pass through before reaching the output.
- A deeper logic chain usually increases the delay and may impact the speed of the circuit.

**Outcome:**
- Trains ML models on a dataset of logic features.
- Compares performance of Random Forest vs Gradient Boosting using metrics like MAE, MSE, and R² score.
- Identifies which circuit features contribute most to logic depth across nodes.
- Generates predictions on new unseen datasets, helping in early design evaluation.

## Code 2 – Propagation Delay Prediction
**Purpose:**
This code predicts the propagation delay of circuits, i.e., how long it takes for a signal to travel from one register to another (register-to-flip-flop delay).
- Propagation delay directly affects the operating frequency of the chip.
- Predicting it early helps in timing closure and avoiding slow designs.

**Outcome:**
- Uses ML regression (Random Forest and Gradient Boosting) to model timing delays.
- Works seamlessly across all technology nodes (14nm–180nm).
- Provides error metrics (MAE, MSE, R²) for accuracy assessment.
- Visualizes predicted vs actual delays to show reliability.
- Exports trained models and predictions for reuse, making it part of a scalable timing analysis pipeline.

## Code 3 – Slack Violation Classification
**Purpose:**
This code checks for timing reliability by predicting whether a circuit path has a slack violation.
- *Slack* = Required Time − *Actual Delay*.
- If *Slack* is negative, the circuit cannot meet its timing requirements → violation.
- If *Slack* is positive, the design is safe.

**Outcome:**
- Optimized primarily for advanced nodes (14nm, 28nm).
- Converts delay data into a binary classification problem: 0 = Safe Path; 1 = Timing Violation
- Handles class imbalance with resampling (important for real-world skewed datasets).
- Evaluates models using Accuracy, Precision, Recall, and F1-score.
- Provides confusion matrices for interpretability.
- Saves trained models and prediction logs, making it useful for circuit reliability verification.

## HOW TO USE
**1. Clone the Repository**
   ```
   git clone https://github.com/RenukaK1404/Machine-Learning-Based-Prediction-of-Combinational-Logic-Depth-Across-Technology-Nodes.git
   cd Machine-Learning-Based-Prediction-of-Combinational-Logic-Depth-Across-Technology-Nodes
   ```
**2. Install dependencies**
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```
**3. Prepare your dataset:**

Ensure to place all these datasets in the same folder as the code
1. combinational_logic_dataset.csv
2. combinational_logic_dataset_predictions.csv
3. combinational_logic_depth_14nm.csv
4. combinational_logic_depth_28nm.csv
5. combinational_logic_depth_45nm.csv
6. combinational_logic_depth_90nm.csv
7. combinational_logic_depth_180nm.csv

**4. Run the scripts**
   - Logic Depth Prediction
     ```
     python Logic-Depth-Prediction.py
     ```
   - Propagation Delay Prediction
     ```
     python Predict-Propagation-Delay.py
     ```
   - Slack Violation Classification
     ```
     python Slack-Violation-Classifier.py
     ```
Each script will:
- Train models automatically.
- Show evaluation metrics & plots.
- Save trained models (.pkl) and prediction results (.csv).

## IN SUMMARY:
- Code 1 **(Logic Depth)** → *How deep is the logic?*
- Code 2 **(Propagation Delay)** → *How long does it take signals to travel?*
- Code 3 **(Slack Violation)** → *Is the circuit safe or unsafe in terms of timing?*

Together, these form a mini end-to-end EDA-inspired ML toolkit for predicting and verifying key performance metrics of combinational circuits.
  
