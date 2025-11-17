# Lap-Time-Prediction-Model
Project that leverages FastF1 to predict next-lap performance at Silverstone 2025 using features such as previous lap times, air and track temperatures, tyre information, and other session metadata.
<img width="1500" height="768" alt="image" src="https://github.com/user-attachments/assets/27c400e9-1b89-41a9-82ff-2bae56191655" />

## Project Description
- Fetches real F1 session data using FastF1  
- Merges car telemetry and weather information  
- Applies feature cleaning and selection  
- Supports both:  
  Global baseline model  
  Driver-specific models  
- Visualizes actual vs predicted lap times for each driver  
- Fully modular project structure  

## Project Structure
```
lap_time_prediction/
    │
    ├── config.py           # Parameters
    ├── load_data.py        # Load & merge laps + weather
    ├── features.py         # Feature engineering + selection utilities
    ├── models.py           # Baseline & driver-specific model training
    ├── visualize.py        # Plotting functions
    ├── main.py             # Main execution pipeline
    ├── requirements.txt    # Dependencies
    └── run.sh              # Execute the entire workflow
```
## How to Run?
### a)
1. Install Requirements
```
pip install -r requirements.txt
```
2. Run Manually
```
python3 main.py
```
### b) 
Run via shell script
```
chmod +x run.sh
./run.sh
```

## Key Approach
1) Data Loading  
- Load lap data using FastF1  
- Load weather data (air temp, humidity, pressure, track temp, wind, etc.)  
- Merge by timestamp  

2) Feature Engineering  
- Encode categorical data (Compound, TrackStatus)  
- Build PrevLapTimeSec grouped by driver (Time Sensitive)  
- Drop direct or overly correlated fields:  
    Sector times  
    Speed traps  
    Session timestamps  
    Pit-in / Pit-out indicators  

3) Feature Selection   
- Compute correlations  
- Select Top 8 numerical features  
- Add essential two categorical features  

4) Two modeling strategies
✅ Baseline Model  
Trained using all drivers mixed, predicts lap time globally.

✅ Driver-specific Models  
Trains one model per driver and evaluates performance individually.

: Driver-specific models can learn each driver’s unique driving style.  
Compare them with the baseline to evaluate whether personalized training improves lap-time prediction performance.

5) Visualization  
- For each driver: Plot actual vs predicted lap times  
- Auto-colored by team livery  
- Titles include driver + team name  











