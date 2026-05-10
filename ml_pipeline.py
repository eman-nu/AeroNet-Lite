import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              accuracy_score, confusion_matrix,
                              classification_report)


# Part A  –  Demand Forecasting
def generate_demand_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset inspired by the Bike Sharing Demand Kaggle competition.
    Features: hour, day_of_week, temperature, weather, zone_density
    Target: demand (deliveries per hour)
    """
    rng = np.random.default_rng(seed)
    hour        = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)
    temperature = rng.uniform(5, 40, n)
    weather     = rng.integers(1, 5, n)      # 1=clear … 4=heavy rain
    zone_density = rng.uniform(500, 8000, n)

    # Demand logic: peaks at rush hours, low at night, drops in bad weather
    base   = 5 + 0.001 * zone_density
    hour_f = np.sin(np.pi * hour / 12) * 8   # sinusoidal hour effect
    temp_f = (temperature - 20) * 0.15
    wx_f   = -(weather - 1) * 1.2
    noise  = rng.normal(0, 1.5, n)
    demand = np.clip(base + hour_f + temp_f + wx_f + noise, 0, None)

    return pd.DataFrame({
        "hour": hour,
        "day_of_week": day_of_week,
        "temperature": temperature,
        "weather": weather,
        "zone_density": zone_density,
        "demand": demand,
    })


def load_demand_data(n_synthetic: int = 800, seed: int = 42) -> pd.DataFrame:
   
    import os
    real_path = os.path.join("data", "train.csv")
    if os.path.exists(real_path):
        print("  [ML] Loading real Bike Sharing Demand dataset ...")
        df = pd.read_csv(real_path, parse_dates=["datetime"])
        df["hour"]         = df["datetime"].dt.hour
        df["day_of_week"]  = df["datetime"].dt.dayofweek
        df["temperature"]  = df["temp"] * 41          # normalised -> approx degC
        df["weather"]      = df["weather"].clip(1, 4)
        df["zone_density"] = 3000                     # placeholder (not in dataset)
        df = df.rename(columns={"count": "demand"})
        print(f"  [ML] Loaded {len(df):,} rows from {real_path}")
        return df[["hour", "day_of_week", "temperature",
                   "weather", "zone_density", "demand"]]
    else:
        print("  [ML] Real dataset not found at data/train.csv")
        print("       Using synthetic data (Bike Sharing Demand distribution).")
        print("       --> Download from https://www.kaggle.com/c/bike-sharing-demand")
        return generate_demand_data(n=n_synthetic, seed=seed)


def run_forecast(verbose: bool = True) -> float:
    """
    Train Linear Regression and Random Forest on demand data.
    Uses real Bike Sharing Demand dataset if available at data/train.csv,
    otherwise falls back to synthetic data.
    Returns MAE of the best model.
    """
    df = load_demand_data()

    features = ["hour", "day_of_week", "temperature", "weather", "zone_density"]
    X = df[features].values
    y = df["demand"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae  = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae  = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    if verbose:
        print("\n" + "="*60)
        print("     DEMAND FORECASTING RESULTS")
        print("="*60)
        print(f"  Model              MAE       RMSE")
        print(f"  ─────────────────────────────────")
        print(f"  Linear Regression  {lr_mae:6.3f}    {lr_rmse:6.3f}")
        print(f"  Random Forest      {rf_mae:6.3f}    {rf_rmse:6.3f}")
        print(f"  Best model: {'Random Forest' if rf_mae < lr_mae else 'Linear Regression'}")
        print("="*60)

    return min(lr_mae, rf_mae)



# Part B  –  Anomaly Detection / Classification
ANOMALY_CLASSES = {0: "Normal", 1: "Battery anomaly",
                   2: "Route anomaly", 3: "Sensor spike"}

def generate_telemetry_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic drone flight telemetry with labelled anomalies.
    Features: battery_drop, speed, route_deviation, altitude_change, speed_change
    Label: 0=Normal, 1=Battery, 2=Route, 3=Sensor
    """
    rng = np.random.default_rng(seed)
    n_per_class = n // 4

    rows = []
    for label in range(4):
        for _ in range(n_per_class):
            if label == 0:   # Normal
                bd  = rng.uniform(1, 5)
                spd = rng.uniform(8, 14)
                dev = rng.uniform(0, 1)
                alt = rng.uniform(-0.5, 0.5)
                sc  = rng.uniform(-1, 1)
            elif label == 1:   # Battery anomaly
                bd  = rng.uniform(18, 40)  # sudden drop
                spd = rng.uniform(6, 12)
                dev = rng.uniform(0, 1.5)
                alt = rng.uniform(-1, 1)
                sc  = rng.uniform(-1, 1)
            elif label == 2:   # Route anomaly
                bd  = rng.uniform(1, 6)
                spd = rng.uniform(8, 14)
                dev = rng.uniform(5, 15)   # large deviation
                alt = rng.uniform(-1, 1)
                sc  = rng.uniform(-1, 1)
            else:              # Sensor spike
                bd  = rng.uniform(1, 6)
                spd = rng.uniform(8, 14)
                dev = rng.uniform(0, 2)
                alt = rng.uniform(10, 40)  # altitude spike
                sc  = rng.uniform(15, 30)  # speed spike
            rows.append([bd, spd, dev, alt, sc, label])

    df = pd.DataFrame(rows,
                      columns=["battery_drop","speed","route_deviation",
                                "altitude_change","speed_change","label"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_anomaly_detection(verbose: bool = True) -> float:
    """
    Train Decision Tree and Random Forest classifiers on telemetry data.
    Returns accuracy of the best model.

    NOTE on 100% accuracy: The synthetic dataset uses clearly separated feature
    ranges per class (e.g. battery_drop > 15 always means Battery anomaly).
    This produces perfect classification scores by design — the goal is to
    demonstrate the pipeline, not to model real-world noise.
    With real UAV telemetry (e.g. CMU ALFA dataset), accuracy would be lower
    due to overlapping features, sensor noise, and edge-case events.
    """
    df = generate_telemetry_data()
    features = ["battery_drop","speed","route_deviation",
                "altitude_change","speed_change"]
    X = df[features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc  = accuracy_score(y_test, dt_pred)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    best_pred  = rf_pred if rf_acc >= dt_acc else dt_pred
    best_label = "Random Forest" if rf_acc >= dt_acc else "Decision Tree"

    if verbose:
        print("\n" + "="*60)
        print("     ANOMALY DETECTION RESULTS")
        print("="*60)
        print(f"  Model              Accuracy")
        print(f"  ──────────────────────────")
        print(f"  Decision Tree      {dt_acc*100:6.2f}%")
        print(f"  Random Forest      {rf_acc*100:6.2f}%")
        print(f"  Best model: {best_label}\n")
        cm = confusion_matrix(y_test, best_pred)
        print("  Confusion Matrix:")
        class_names = [ANOMALY_CLASSES[i] for i in range(4)]
        header = "  " + " ".join(f"{c[:7]:>8}" for c in class_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = "  " + f"{class_names[i][:7]:>8}" + " ".join(f"{v:8d}" for v in row)
            print(row_str)
        print()
        print("  Classification Report:")
        print(classification_report(y_test, best_pred,
                                    target_names=list(ANOMALY_CLASSES.values()),
                                    zero_division=0))
        print("="*60)

    return max(dt_acc, rf_acc)


def classify_drone_reading(battery_drop, speed, route_deviation,
                            altitude_change, speed_change) -> str:
    """
    Quick in-memory classifier for live anomaly detection during simulation.
    Uses simple rule-based fallback (same logic as synthetic data generation).
    """
    if battery_drop > 15:
        return "Battery anomaly"
    if route_deviation > 4:
        return "Route anomaly"
    if abs(altitude_change) > 8 or abs(speed_change) > 12:
        return "Sensor spike"
    return "Normal"


if __name__ == "__main__":
    run_forecast()
    run_anomaly_detection()
