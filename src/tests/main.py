from env import PseudoLabelEnv
from model import RLmodel

def main():
    env = PseudoLabelEnv()
    model = RLmodel()
    state, info = env.reset()
    done = False

    while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        model.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated

    env.close()

if __name__ == "main":
    main()

