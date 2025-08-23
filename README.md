# Machine-Learning-Based-Prediction-of-Combinational-Logic-Depth-Across-Technology-Nodes
Machine learning-based prediction of combinational logic depth across multiple semiconductor technology nodes (14nm–180nm). The project applies data-driven models to estimate circuit depth, enabling faster, more efficient VLSI design optimization.

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
- Identifies which circuit features contribute most to logic depth across 14nm → 180nm nodes.
- Generates predictions on new unseen datasets, helping in early design evaluation.

## Code 2 – Propagation Delay Prediction
**Purpose:**
This code predicts the propagation delay of circuits, i.e., how long it takes for a signal to travel from one register to another (register-to-flip-flop delay).
- Propagation delay directly affects the operating frequency of the chip.
- Predicting it early helps in timing closure and avoiding slow designs.

**Outcome:**
- Uses ML regression (Random Forest and Gradient Boosting) to model timing delays.
- Provides error metrics (MAE, MSE, R²) for accuracy assessment.
- Visualizes predicted vs actual delays to show reliability.
- Exports trained models and predictions for reuse, making it part of a scalable timing analysis pipeline.

## Code 3 – Slack Violation Classification
**Purpose:** Detect whether paths meet timing constraints.
**Outcome:**
- Binary classifier: Safe (0) / Violation (1).
- Handles imbalance via resampling.
- Confusion matrix + Precision/Recall analysis.
**Purpose:**
This code checks for timing reliability by predicting whether a circuit path has a slack violation.
- *Slack* = Required Time − *Actual Delay*.
- If *Slack* is negative, the circuit cannot meet its timing requirements → violation.
- If *Slack* is positive, the design is safe.

**Outcome:**
- Converts delay data into a binary classification problem:
  ```
  0 = Safe Path
  1 = Timing Violation
  ```
- Handles class imbalance with resampling (important for real-world skewed datasets).
- Evaluates models using Accuracy, Precision, Recall, and F1-score.
- Provides confusion matrices for interpretability.
- Saves trained models and prediction logs, making it useful for circuit reliability verification.

**In summary:**
- Code 1 **(Logic Depth)** → *How deep is the logic?*
- Code 2 **(Propagation Delay)** → *How long does it take signals to travel?*
- Code 3 **(Slack Violation)** → *Is the circuit safe or unsafe in terms of timing?*

Together, these form a mini end-to-end EDA-inspired ML toolkit for predicting and verifying key performance metrics of combinational circuits.
  
