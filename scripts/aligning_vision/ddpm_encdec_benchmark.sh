MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_encdec_vision_agent \
              agent_name=ddpm_encdec_vision \
              window_size=8 \
              group=ddpm_encdec_vision_seeds
