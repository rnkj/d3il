MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
    --multirun seed=0,1,2,3,4,5 \
    agents=mingru_vision_agent \
    agent_name=mingru_vision \
    group=aligning_mingru_seeds \
    window_size=5