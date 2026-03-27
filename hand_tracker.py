import cv2
import mediapipe as mp
import socket

# ==========================================
# 1. NETWORK CONFIGURATION
# ==========================================
UDP_IP = "127.0.0.1"  # Localhost
UDP_PORT = 9000       # Must match the Blender listening port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==========================================
# 2. MEDIAPIPE INITIALIZATION
# ==========================================
mp_hands = mp.solutions.hands
# max_num_hands=1 ensures we don't send conflicting data if two hands are visible
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the primary webcam
cap = cv2.VideoCapture(0)

# ==========================================
# 3. KINEMATIC MAPPING FUNCTION
# ==========================================
def map_range(value, in_min, in_max, out_min, out_max):
    """Proportionally scales a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

print("Initializing Vision System. Press 'q' in the video window to quit.")

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture video feed.")
        break
        
    # Flip the image horizontally for an intuitive "selfie" mirror view
    img = cv2.flip(img, 1)
    
    # MediaPipe requires RGB color space, but OpenCV captures in BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Extract Landmark 9: Middle Finger MCP (Knuckle)
            tracker_node = hand_landmarks.landmark[9]
            
            # Calculate Pan (X-Axis): Left/Right
            # Camera X goes 0.0 (Left) to 1.0 (Right). We map this to -45 to 45 degrees.
            pan_angle = map_range(tracker_node.x, 0.0, 1.0, -80.0, 80.0)
            
            # Calculate Tilt (Y-Axis): Up/Down
            # Camera Y goes 0.0 (Top) to 1.0 (Bottom). 
            # We map 0.0 (Top) to 30 degrees, and 1.0 (Bottom) to -30 degrees to invert the axis naturally.
            tilt_angle = map_range(tracker_node.y, 0.0, 1.0, 60.0, -60.0)
            
            # Format and Broadcast the UDP Data
            message = f"{pan_angle},{tilt_angle}"
            
            # ADD THIS LINE: Print the angles to the terminal for debugging
            print(f"Broadcasting to Port {UDP_PORT} -> Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            
            # Draw the neural network mesh over the hand for visual debugging
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the live feed
    cv2.imshow("Digital Twin - Spatial Tracking", img)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()