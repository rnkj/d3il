python run.py --config-name=stacking_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_encdec \
              agent_name=ddpm_encdec \
              window_size=5 \
              group=stacking_ddpm_encdec_h5a3 \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=18 \
              agents.action_seq_size=3 \
              agents.obs_seq_len=1 \
              agents.model.n_timesteps=16 \
              agents.model.model.encoder.embed_dim=120 \
              agents.model.model.decoder.embed_dim=120 \
              agents.model.model.action_seq_len=5 \
              agents.model.model.obs_seq_len=5