MUJOCO_GL=egl python run_vision.py --config-name=stacking_vit_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=dinov2_bc_vision_agent \
              agent_name=dinov2_bc_vision \
              window_size=1 \
              group=stacking_bc_seeds