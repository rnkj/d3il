MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
    --multirun seed=0,1,2,3,4,5 \
    agents=lstm_vision_agent \
    agent_name=lstm_vision \
    group=aligning_lstm_seeds \
    window_size=5