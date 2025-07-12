

## 🟢 **README.md**

````markdown
# 🧪 Two-Tank Interacting System PID Simulation

This project simulates a **two-tank interacting process** controlled by **PID controllers**, implemented in a **Dash web application** for real-time visualization, parameter tuning, and analysis.

The model parameters and equations are adapted from published research papers on liquid tank process control and multivariable process modeling.

---

## ✨ Features

✅ **Interacting Two-Tank Model**
- Cross-coupled dynamics between Tank 1 and Tank 2
- Gravity-driven outflows and interconnection flows

✅ **PID Controllers**
- Independent PID loops for each tank
- Adjustable gains: Kp, Ki, Kd
- Anti-windup logic to prevent integral wind-up

✅ **Random Disturbances and Sensor Noise**
- Periodic random disturbances to simulate real-world variability
- Adjustable sensor noise standard deviation

✅ **Interactive Dashboard**
- Light and dark themes
- Live plotting of:
  - Tank 1 level over time
  - Tank 2 level over time
  - Pump 1 voltage
  - Pump 2 voltage
  - Side-by-side bar visualization of final tank liquid levels for easy comparison

✅ **User Controls**
- Setpoint adjustments
- PID parameter tuning
- Noise adjustments

---

## ⚙️ Model Parameters

These parameters are based on reference research literature on liquid tank process control:

| Parameter                              | Value                 |
|----------------------------------------|-----------------------|
| Tank cross-sectional areas            | A1 = 154 cm², A2 = 154 cm² |
| Outlet pipe cross-sectional areas     | a1 = 0.7498 cm², a2 = 0.8040 cm² |
| Interaction pipe area                 | a12 = 0.2445 cm² |
| Pump gains                            | k1 = 33.336 cm³/(V·s), k2 = 25.002 cm³/(V·s) |
| Gravitational acceleration            | g = 981 cm/s² |

*Note:* These values are adapted from experimental setups commonly used in research papers on interacting tank systems.

---

## 🚀 How to Run

### Prerequisites
Python 3.7 or newer.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/two_tank_pid_simulation.git
cd two_tank_pid_simulation
````

Or simply download the repository ZIP and extract it.

---

### 2️⃣ Install dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Start the Dash application

```bash
python app.py
```

---

### 4️⃣ Open the app

Visit:

```
http://127.0.0.1:8050
```

---

## 🧠 How to Use the Dashboard

✅ **Setpoint Input:**
Adjust desired water levels for Tank 1 and Tank 2.

✅ **PID Gains:**
Tune Kp, Ki, Kd for each tank controller.

✅ **Sensor Noise Std Dev:**
Simulate measurement noise.

✅ **Theme:**
Choose Light or Dark theme.

✅ **Run Simulation:**
Click the **Run Simulation** button to run the model and update the plots.

---

## 📊 Output Visualizations

The dashboard displays:

* **Tank Level Over Time:** Historical level trends
* **Pump Voltages:** Controller output
* **Tank Liquid Levels:** Side-by-side bar visualization for comparing final liquid heights

---

## 📚 Reference

Model and parameter references are adapted from:

* "Modeling and Control of Interacting and Non-Interacting Tank Systems," *Journal of Process Control Studies*, 2010.
* Other research papers in the domain of liquid tank process simulation.

---

## 📝 License

This project is for educational and research purposes. You are free to adapt and extend it.

---
