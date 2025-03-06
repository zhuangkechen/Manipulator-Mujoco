import gymnasium
import manipulator_mujoco
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def get_position(hand_landmarks, hand_index=mp_hands.HandLandmark.WRIST):
    wrist = hand_landmarks.landmark[hand_index]
    # 根据需要调整尺度和偏移
    x, y, z = -wrist.x+0.5, -wrist.y+0.5, wrist.z * (10**5)
    return np.array([x, y, z])

def get_orientation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # 计算方向向量
    direction = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y, index_tip.z - wrist.z])
    direction = direction / np.linalg.norm(direction)  # 归一化
    
    # 假设初始朝向为 (1, 0, 0)，计算需要的旋转四元数
    # 这里仅为示例，实际应用中需要更精确的计算
    axis = np.cross([1, 0, 0], direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot([1, 0, 0], direction))
    s = np.sin(angle / 2)
    qx, qy, qz = axis * s
    qw = np.cos(angle / 2)
    
    return np.array([qx, qy, qz, qw])

def calculate_jaw_control(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    
    # 计算归一化距离
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + 
                       (thumb_tip.y - index_tip.y) ** 2 + 
                       (thumb_tip.z - index_tip.z) ** 2)
    
    # 将距离映射到 -3.14 到 3.14 范围
    # 假设最大距离为 0.2，这个值可以根据实际情况调整
    max_distance = 0.2
    control_value = (distance / max_distance) * 3.14 * 2 - 3.14
    
    return max(-0.5, min(3.14, control_value))

def get_action_from_gesture(frame, hands):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    action = np.zeros(8)  # 对应于动作空间的7个维度

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            p = get_position(hand_landmarks)
            q = get_orientation(hand_landmarks)
            a = calculate_jaw_control(hand_landmarks)

            # 空间坐标
            action[:3] = p
            # 手掌旋转（简化示例）
            action[3:7] = q
            # 抓取动作
            action[7] = a
    return action

# Create the environment with rendering in human mode
env = gymnasium.make('manipulator_mujoco/SO100Env-v0', render_mode='human')

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)


cap = cv2.VideoCapture(0)

# Run simulation for a fixed number of steps
# for _ in range(1000):
count_frame = 0
frame_skip = 10
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count_frame % (frame_skip + 1) == 0:
        action = get_action_from_gesture(frame, hands)
    else:
        action = None
    count_frame += 1
    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # Choose a random action from the available action space
    action = env.action_space.sample()

    # Take a step in the environment using the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the episode is over (terminated) or max steps reached (truncated)
    if terminated or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset()
    # cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the environment when the simulation is done
env.close()
cap.release()
cv2.destroyAllWindows()