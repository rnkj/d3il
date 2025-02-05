MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
    agents=sarnn_agent \
    agent_name=sarnn \
    group=sorting_4_sarnn_seeds \
    train_batch_size=4 \
    val_batch_size=4 \
    epoch=10000 \
    eval_every_n_epochs=500
