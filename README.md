# MindTune: EEG Feature Engineering & Transformation Pipeline

This README document outlines the mathematical and neurological framework used to transform raw EEG signals into a high-accuracy (91%+) feature vector for the **Extra Trees** classification model.

---

## 🧠 The Core Problem
Raw EEG data is inherently "noisy" and cannot be used for machine learning without three critical transformations:
1.  **Scale Variance**: Raw values differ based on skin conductance and hardware, making users "incomparable".
2.  **Lack of Context**: Models require "domain knowledge" to understand known neural stress markers.
3.  **Temporal Noise**: Single-second snapshots are too volatile; stress is a state that builds over time.

---

## 🛠 Layer 1: Band Power Normalization
To solve the inter-subject scale problem, we convert absolute power into **scale-invariant proportions**[cite: 31, 137].

### The Math
For each of the 8 primary bands, we calculate its percentage of the total spectral energy[cite: 22, 24]:
$$\text{Band}_i \text{ (pct)} = \frac{\text{Band}_i}{\sum_{j=1}^{8} \text{Band}_j + \epsilon}$$
*(An epsilon $\epsilon = 1e-6$ is added to prevent division by zero[cite: 38].)*

### Frequency Band Mapping [cite: 12]
| Band | Range | Brain Region | Mental State | Stress Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Delta ($\delta$)** | 0.5–4 Hz | Occipital | Deep sleep | ↓ when stressed |
| **Theta ($\theta$)** | 4–8 Hz | Temporal/Frontal | Meditative | $\theta/\beta \uparrow$ = stressed |
| **Alpha ($\alpha$)** | 8–12 Hz | Parietal | Relaxed | ↓ during stress |
| **Beta ($\beta$)** | 12–30 Hz | Frontal | Focused/Alert | $\uparrow$ during stress |
| **Gamma ($\gamma$)** | 30–40+ Hz | Occipital | High Load | $\uparrow$ under stress |

---

## 🧪 Layer 2: Engineered Cognitive Ratios
We "inject" 50+ years of neuroscience research by pre-computing clinical stress biomarkers[cite: 35, 139].

* **Theta-Beta Ratio (TBR)**: Measures cortical arousal[cite: 41, 42].
    * *High Ratio*: Calm/Drowsy[cite: 47].
    * *Low Ratio*: Anxious/Aroused (Stress Indicator)[cite: 48].
* **Alpha-Beta Ratio (ABR)**: Captures the balance between idling and active cognition[cite: 49, 52].
    * *High Ratio*: Relaxed/Resting[cite: 56].
    * *Low Ratio*: Aroused/Stressed[cite: 57].
* **Slow-Fast Ratio (SFR)**: $(\delta + \theta) / (\beta + \gamma)$. A broad measure of cognitive load[cite: 60, 63].

---

## ⏱ Layer 3: Temporal Rolling Statistics
To remove "snapshot noise," we use a **FIFO Buffer (deque maxlen=5)** to capture the last 5 seconds of brain activity[cite: 73, 77].



1.  **Rolling Mean ($\mu$):** Captures **Trend**. Tells the model if a state (like Beta) is rising or falling over time[cite: 98, 99].
2.  **Rolling Std Dev ($\sigma$):** Captures **Stability**. High volatility (high std) is a strong discriminator for unstable, stressed brain states[cite: 107, 108].

---

## 📊 Final Feature Vector (30+ Total)
The pipeline outputs a comprehensive vector every second[cite: 135]:

| Group | Features | Why? |
| :--- | :--- | :--- |
| **Band Pct** | 8 (Delta through Gamma) | Scale-invariant proportions[cite: 135]. |
| **Ratios** | 3 (TBR, ABR, SFR) | Neuroscience prior knowledge[cite: 135]. |
| **Rolling Mean** | 8 (per band) | Temporal trend direction[cite: 135]. |
| **Rolling Std** | 8 (per band) | Brain state volatility[cite: 135]. |
| **Base Metrics** | 3 (Attention, Meditation, Quality) | Proprietary hardware scores[cite: 135]. |

> **Note**: The system requires a 5-second "warm-up" period to fully populate the buffer before valid predictions can begin[cite: 132, 133].
