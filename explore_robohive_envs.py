import torch
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from torchrl.envs import RoboHiveEnv
from torchrl.envs.transforms import TransformedEnv

ENV_NAMES_TO_EXPLORE = [
    "DKittyStandRandom-v0",
    "FetchReachRandom-v0",
    "FK1_LdoorOpenRandom-v4",
    "FrankaPushRandom-v0",
    "MyoHandCupDrink-v0",
    "relocate-v1",
    "rpFrankaRobotiqData-v0"
]


def main():
    print(f"Envs : {RoboHiveEnv.available_envs}")
    envs = []
    for env_name in ENV_NAMES_TO_EXPLORE:
        print(f"Exploring {env_name}...")

        video_logger = CSVLogger(exp_name=env_name, log_dir="videos", video_format="mp4", video_fps=30)
        recorder = VideoRecorder(logger=video_logger, tag="iteration", skip=2)

        env = TransformedEnv(
            RoboHiveEnv(env_name=env_name,
                        from_pixels=True,
                        pixels_only=True,
                        from_depths=False,
                        frame_skip=None)
        )
        env.append_transform(recorder)
        envs.append(env)

        policy = RandomPolicy(action_spec=env.action_spec)

        device = torch.device('cpu')

        collector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            total_frames=200,
            max_frames_per_traj=50,
            frames_per_batch=200,
            device=device,
            storing_device=device,
        )

        for _ in collector:
            continue

        env.transform.dump()
    
    for env in envs:
        env.close()

if __name__ == "__main__":
    main()