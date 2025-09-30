import numpy as np
import matplotlib.pyplot as plt
from crazy_flie_env import CrazyFlieEnv, EnvConfig
from pynput import keyboard
import threading
import time

# --- Keyboard control mapping ---
# You may need to adjust this mapping based on your action space definition
# Example: [thrust, roll, pitch, yaw]
ACTION_SIZE = 4  # Change if your action space shape is different
ACTION_STEP = 0.1

key_to_action = {
    'a':     (0, -ACTION_STEP),  # roll left
    'd':     (0, +ACTION_STEP),  # roll right
    'up':    (1, +ACTION_STEP),  # pitch forward
    'down':  (1, -ACTION_STEP),  # pitch back
    'left':  (2, -ACTION_STEP),  # yaw left
    'right': (2, +ACTION_STEP),  # yaw right
    'w':     (3, +ACTION_STEP),  # thrust up
    's':     (3, -ACTION_STEP),  # thrust down
}

def keyboard_control(action, stop_flag):
    def on_press(key):
        try:
            k = key.char.lower()
        except:
            k = key.name
        if k in key_to_action:
            idx, delta = key_to_action[k]
            action[idx] += delta
        if k == 'esc':
            stop_flag[0] = True
            return False
    def on_release(key):
        pass  # Optionally, you can zero out action on release
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    config = EnvConfig()
    env = CrazyFlieEnv(config)
    env.renderer.set_room_transparency(env.physics.model, alpha=0.3)
    obs, info = env.reset()
    done = False
    step_count = 0
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    stop_flag = [False]

    # Start keyboard listener in a separate thread
    t = threading.Thread(target=keyboard_control, args=(action, stop_flag), daemon=True)
    t.start()

    print("\nControl the drone with arrow keys and WASD. Press ESC to quit.")
    try:
        while not stop_flag[0]:
            env.render()
            obs, reward, terminated, truncated, info = env.step(action.copy())
            state = obs['state']
            x, y, z = state[0:3]
            vx, vy, vz = state[3:6]
            roll, pitch, yaw = state[6:9]
            wx, wy, wz = state[9:12]
            print(f"Step {step_count}: action={action}, reward={reward:.3f}, terminated={terminated}, truncated={truncated} | "
                  f"pos=({x:.2f}, {y:.2f}, {z:.2f}), vel=({vx:.2f}, {vy:.2f}, {vz:.2f}), "
                  f"orient=(roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}), ang_vel=({wx:.2f}, {wy:.2f}, {wz:.2f})")
            step_count += 1
            if terminated or truncated:
                print("Drone crashed or episode ended. Respawning...")
                obs, info = env.reset()
                action[:] = 0  # Optionally reset action to zero
                continue
            time.sleep(0.05)  # Control frequency
    finally:
        env.close()
        print("Exited drone control loop.")

if __name__ == "__main__":
    main()
