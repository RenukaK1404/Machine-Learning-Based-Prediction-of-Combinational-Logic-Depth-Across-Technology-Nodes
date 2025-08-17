# Machine-Learning-Based-Prediction-of-Combinational-Logic-Depth-Across-Technology-Nodes
Machine learning-based prediction of combinational logic depth across multiple semiconductor technology nodes (14nm–180nm). The project applies data-driven models to estimate circuit depth, enabling faster, more efficient VLSI design optimization.

## INTRODUCTION
Every digital device you use—your phone, laptop, or smartwatch—runs on **logic circuits** built from simple gates like AND, OR, and NOT.  
One key property of these circuits is **logic depth**, which tells us **how many layers of gates a signal must pass through** before producing an output.  

- A **small depth** → faster circuits, lower power consumption, better efficiency.  
- A **large depth** → slower circuits, higher energy use, possible timing failures.  

As semiconductor technology scales down (from **180nm → 14nm chips**), circuits become more complex and harder to analyze.  
Traditionally, engineers estimate logic depth and timing through **manual simulations**, which are slow and resource-intensive.  

👉 This project applies **Machine Learning (ML)** to predict:  
- **Logic Depth** (circuit complexity)  
- **Propagation Delay** (signal travel time)  
- **Slack Violations** (timing safety check)  

This makes the design process **faster, smarter, and more automated**, helping engineers build efficient chips and paving the way for smarter applications—even in fields like **biomedical device circuits**.  

## PROJECT OBJECTIVE  
- Develop ML models that:  
  ✅ Predict **logic depth** across multiple technology nodes.  
  ✅ Estimate **propagation delays** (timing analysis).  
  ✅ Classify **slack violations** (timing reliability check).

## Code 1 – Logic Depth Prediction
**Purpose:** Estimate how deep the logic chain is.
**Outcome:**
- Regression models (RF, GBM).
- Feature importance ranking.
- Predicts depth across 14nm → 180nm nodes.

## Code 2 – Propagation Delay Prediction
**Purpose:** Estimate how long signals take between registers.
**Outcome:**
- Accurate delay regression.
- Visualizes predicted vs. actual delay.
- Early insights for timing closure.

## Code 3 – Slack Violation Classification
**Purpose:** Detect whether paths meet timing constraints.
**Outcome:**
- Binary classifier: Safe (0) / Violation (1).
- Handles imbalance via resampling.
- Confusion matrix + Precision/Recall analysis.
  
