MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
    agents=sarnn_agent \
    agent_name=sarnn \
    group=sorting_4_sarnn_seeds \
    train_batch_size=2 \
    val_batch_size=2 \
    epoch=1000
