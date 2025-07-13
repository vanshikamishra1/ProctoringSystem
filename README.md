# AI-Powered Proctoring System

An intelligent computer vision-based proctoring system that monitors students during online exams using real-time head pose estimation, eye gaze tracking, and face detection to detect potential cheating behaviors.

## 🎯 Features

### Real-time Monitoring
- **Head Pose Estimation**: Tracks pitch, yaw, and roll angles of the head
- **Eye Gaze Tracking**: Monitors eye movement and detects when eyes are looking down
- **Face Detection**: Ensures only one person is present in the camera view
- **Multi-face Detection**: Alerts when multiple faces are detected simultaneously

### Cheating Detection
- **Head Turning**: Detects when head is turned beyond acceptable thresholds
- **Looking Down**: Identifies when eyes are focused downward (potential phone/notes usage)
- **Face Absence**: Alerts when no face is detected in the camera view
- **Multiple Persons**: Detects presence of multiple individuals

### Analytics & Reporting
- **Real-time Statistics**: Displays cheating events, duration, and attentive accuracy
- **CSV Logging**: Comprehensive timestamped logs of all monitoring data
- **Screenshot Capture**: Automatically saves images when cheating is detected
- **Session Recording**: Records the entire proctoring session as video

## 📋 Requirements

### System Requirements
- Windows 10/11
- Webcam (built-in or external)
- Python 3.7 or higher

### Python Dependencies
```
opencv-python
mediapipe
numpy
```

## 🚀 Installation

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd ProctoringProject
   ```

2. **Install Python dependencies**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Run the application**
   - **Option 1**: Double-click `run_proctoring.bat`
   - **Option 2**: Run directly with Python
     ```bash
     python proctor_gui_terminal.py
     ```

## 🎮 Usage

### Starting the Proctoring Session
1. Ensure your webcam is connected and working
2. Run the application using one of the methods above
3. Position yourself in front of the camera
4. The system will automatically start monitoring

### Real-time Display
The application shows a live video feed with overlay information:
- **Pitch**: Vertical head movement (looking up/down)
- **Yaw**: Horizontal head movement (turning left/right)
- **Gaze**: Eye position ratio (detects looking down)
- **Face**: Whether a face is currently detected
- **Events**: Number of cheating incidents detected
- **Cheating Time**: Total duration of cheating behaviors
- **Attentive Accuracy**: Percentage of time spent being attentive

### Controls
- **ESC Key**: Exit the proctoring session
- **Close Window**: Click the X button to stop monitoring

## 📊 Output Files

### Generated Files
- `head_pose_log.csv`: Detailed CSV log with timestamps and all monitoring data
- `session_recording.avi`: Complete video recording of the proctoring session
- `cheating_screenshots/`: Folder containing images captured during cheating incidents

### CSV Log Format
| Column | Description |
|--------|-------------|
| Timestamp | Date and time of the measurement |
| Pitch | Vertical head angle (degrees) |
| Yaw | Horizontal head angle (degrees) |
| Roll | Head rotation angle (degrees) |
| Cheating Detected | Yes/No flag for cheating detection |
| Reason | Specific reason for cheating alert |

## ⚙️ Configuration

### Thresholds (in `proctor_gui_terminal.py`)
```python
PITCH_THRESHOLD = 20      # Maximum allowed pitch angle
YAW_THRESHOLD = 30        # Maximum allowed yaw angle
EYE_DOWN_RATIO_THRESHOLD = 0.65  # Eye gaze threshold for looking down
```

### Adjusting Sensitivity
- **Lower thresholds**: More sensitive detection (more false positives)
- **Higher thresholds**: Less sensitive detection (may miss some behaviors)

## 🔧 Technical Details

### Computer Vision Pipeline
1. **Face Detection**: Uses MediaPipe Face Mesh for precise facial landmark detection
2. **Head Pose Estimation**: Implements solvePnP algorithm with 6-point model
3. **Eye Gaze Analysis**: Calculates iris position relative to eye boundaries
4. **Real-time Processing**: Processes video frames at ~20 FPS

### Detection Algorithms
- **Head Pose**: 3D rotation estimation using facial landmarks
- **Eye Gaze**: Ratio-based calculation of iris position within eye socket
- **Multi-face**: Counts detected faces using MediaPipe

## 📈 Performance Metrics

### Attentive Accuracy
- Calculated as: `(attentive_time / total_session_time) × 100`
- Real-time display shows current accuracy percentage
- Helps assess overall session compliance

### Cheating Statistics
- **Event Counter**: Number of distinct cheating incidents
- **Duration Tracking**: Total time spent in cheating behaviors
- **Real-time Alerts**: Immediate visual warnings during incidents

## 🛡️ Privacy & Ethics

### Data Handling
- All data is stored locally on the user's machine
- No data is transmitted to external servers
- Screenshots and recordings are for review purposes only

### Recommended Usage
- Inform students about monitoring before exams
- Use as a supplement to traditional proctoring methods
- Review flagged incidents manually before taking action
- Consider false positives in assessment decisions

## 🐛 Troubleshooting

### Common Issues

**Camera not detected**
- Ensure webcam is connected and not in use by other applications
- Check camera permissions in Windows settings

**Poor face detection**
- Ensure good lighting conditions
- Position face clearly in camera view
- Remove obstructions (glasses, masks, etc.)

**High false positive rate**
- Adjust threshold values in the code
- Ensure stable camera positioning
- Check for lighting changes during session

### Performance Optimization
- Close unnecessary applications to free up CPU resources
- Use a dedicated webcam for better performance
- Ensure adequate lighting for optimal face detection

## 📝 License

This project is provided as-is for educational and research purposes. Please ensure compliance with local privacy laws and institutional policies when using this system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the system.

## 📞 Support

For technical support or questions about implementation, please refer to the code comments or create an issue in the project repository. 