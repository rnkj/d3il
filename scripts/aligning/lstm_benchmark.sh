python run.py --config-name=aligning_config \
    --multirun seed=0,1,2,3,4,5 \
    agents=lstm_agent \
    agent_name=lstm \
    window_size=3 \
    group=aligning_lstm_seeds