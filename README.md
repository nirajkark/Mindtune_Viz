# MindTune: EEG Feature Engineering & Transformation Pipeline

[cite_start]This README document outlines the mathematical and neurological framework used to transform raw EEG signals into a high-accuracy (91%+) feature vector for the **Extra Trees** classification model[cite: 2, 152].

---

## 🧠 The Core Problem
[cite_start]Raw EEG data is inherently "noisy" and cannot be used for machine learning without three critical transformations[cite: 5, 6]:
1.  [cite_start]**Scale Variance**: Raw values differ based on skin conductance and hardware, making users "incomparable"[cite: 7, 13].
2.  [cite_start]**Lack of Context**: Models require "domain knowledge" to understand known neural stress markers[cite: 7, 36].
3.  [cite_start]**Temporal Noise**: Single-second snapshots are too volatile; stress is a state that builds over time[cite: 7, 74].

---

## 🛠 Layer 1: Band Power Normalization
[cite_start]To solve the inter-subject scale problem, we convert absolute power into **scale-invariant proportions**[cite: 31, 137].

### The Math
[cite_start]For each of the 8 primary bands, we calculate its percentage of the total spectral energy[cite: 22, 24]:
$$\text{Band}_i \text{ (pct)} = \frac{\text{Band}_i}{\sum_{j=1}^{8} \text{Band}_j + \epsilon}$$
[cite_start]*(An epsilon $\epsilon = 1e-6$ is added to prevent division by zero[cite: 38].)*

### [cite_start]Frequency Band Mapping [cite: 12]
| Band | Range | Brain Region | Mental State | Stress Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Delta ($\delta$)** | 0.5–4 Hz | Occipital | Deep sleep | ↓ when stressed |
| **Theta ($\theta$)** | 4–8 Hz | Temporal/Frontal | Meditative | $\theta/\beta \uparrow$ = stressed |
| **Alpha ($\alpha$)** | 8–12 Hz | Parietal | Relaxed | ↓ during stress |
| **Beta ($\beta$)** | 12–30 Hz | Frontal | Focused/Alert | $\uparrow$ during stress |
| **Gamma ($\gamma$)** | 30–40+ Hz | Occipital | High Load | $\uparrow$ under stress |

---

## 🧪 Layer 2: Engineered Cognitive Ratios
[cite_start]We "inject" 50+ years of neuroscience research by pre-computing clinical stress biomarkers[cite: 35, 139].

* [cite_start]**Theta-Beta Ratio (TBR)**: Measures cortical arousal[cite: 41, 42].
    * [cite_start]*High Ratio*: Calm/Drowsy[cite: 47].
    * [cite_start]*Low Ratio*: Anxious/Aroused (Stress Indicator)[cite: 48].
* [cite_start]**Alpha-Beta Ratio (ABR)**: Captures the balance between idling and active cognition[cite: 49, 52].
    * [cite_start]*High Ratio*: Relaxed/Resting[cite: 56].
    * [cite_start]*Low Ratio*: Aroused/Stressed[cite: 57].
* **Slow-Fast Ratio (SFR)**: $(\delta + \theta) / (\beta + \gamma)$. [cite_start]A broad measure of cognitive load[cite: 60, 63].

---

## ⏱ Layer 3: Temporal Rolling Statistics
[cite_start]To remove "snapshot noise," we use a **FIFO Buffer (deque maxlen=5)** to capture the last 5 seconds of brain activity[cite: 73, 77].



1.  **Rolling Mean ($\mu$):** Captures **Trend**. [cite_start]Tells the model if a state (like Beta) is rising or falling over time[cite: 98, 99].
2.  **Rolling Std Dev ($\sigma$):** Captures **Stability**. [cite_start]High volatility (high std) is a strong discriminator for unstable, stressed brain states[cite: 107, 108].

---

## 📊 Final Feature Vector (30+ Total)
[cite_start]The pipeline outputs a comprehensive vector every second[cite: 135]:

| Group | Features | Why? |
| :--- | :--- | :--- |
| **Band Pct** | 8 (Delta through Gamma) | [cite_start]Scale-invariant proportions[cite: 135]. |
| **Ratios** | 3 (TBR, ABR, SFR) | [cite_start]Neuroscience prior knowledge[cite: 135]. |
| **Rolling Mean** | 8 (per band) | [cite_start]Temporal trend direction[cite: 135]. |
| **Rolling Std** | 8 (per band) | [cite_start]Brain state volatility[cite: 135]. |
| **Base Metrics** | 3 (Attention, Meditation, Quality) | [cite_start]Proprietary hardware scores[cite: 135]. |

> [cite_start]**Note**: The system requires a 5-second "warm-up" period to fully populate the buffer before valid predictions can begin[cite: 132, 133].
