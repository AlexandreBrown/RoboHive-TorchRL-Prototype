import torch
import numpy as np
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from torchrl.envs import RoboHiveEnv
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs import EnvBase


class VisualDomainRandomizedEnv(EnvBase):
    def __init__(self, env, cameras: list = ["top_cam"]):
        super().__init__(
            device=env.device, batch_size=env.batch_size, allow_done_after_reset=False
        )
        self._base_env = env
        self.cameras = cameras
        self.observation_spec = env.observation_spec.clone()
        self.action_spec = env.action_spec.clone()

    def _step(self, tensordict):
        # self.randomize_visual()
        tensordict = self._base_env._step(tensordict)

        # obs = self.get_sim_observation()
        # tensordict.set("observation", obs)

        return tensordict

    def apply_visual_domain_randomization(self) -> torch.Tensor:

        # Randomize geometry color
        self._base_env.sim.model.geom_rgba[:] = np.random.rand(
            self._base_env.sim.model.ngeom, 4
        )

        # Randomize geometry size
        # self._base_env.sim.model.geom_size[:] = np.random.uniform(low=0.8, high=1.2, size=self._base_env.sim.model.geom_size.shape) * self._base_env.sim.model.geom_size

        # Randomize geometry positions
        # self._base_env.sim.model.geom_pos[:] += np.random.normal(0, 0.01, self._base_env.sim.model.geom_pos.shape)

        # Randomize geometry orientation
        # self._base_env.sim.model.geom_quat[:] = np.random.randn(self._base_env.sim.model.ngeom, 4)
        # self._base_env.sim.model.geom_quat[:] = self._base_env.sim.model.geom_quat[:] / np.linalg.norm(self._base_env.sim.model.geom_quat[:], axis=-1, keepdims=True)

        # Randomize material textures
        # self._base_env.sim.model.mat_texid[:] = np.random.randint(low=0, high=self._base_env.sim.model.ntex, size=self._base_env.sim.model.mat_texid.shape)

        # Randomize material colors
        # self._base_env.sim.model.mat_rgba[:] = np.random.rand(self._base_env.sim.model.nmat, 4)

        # Randomize material shininess
        # self._base_env.sim.model.mat_shininess[:] = np.random.uniform(0, 1, size=self._base_env.sim.model.mat_shininess.shape)
        # self._base_env.sim.model.mat_specular[:] = np.random.uniform(0, 1, size=self._base_env.sim.model.mat_specular.shape)
        # self._base_env.sim.model.mat_emission[:] = np.random.uniform(0, 1, size=self._base_env.sim.model.mat_emission.shape)

        # Randomize light diffuse
        # self._base_env.sim.model.light_diffuse[:] = np.random.rand(3)

        # Randomize ambiant lighting
        # self._base_env.sim.model.light_ambient[:] = np.random.rand(self._base_env.sim.model.nlight, 3)

        # Randomize light position
        # self._base_env.sim.model.light_pos[:] += np.random.normal(0, 0.1, size=self._base_env.sim.model.light_pos.shape)

        # Randomize light direction
        # self._base_env.sim.model.light_dir[:] += np.random.normal(0, 0.1, size=self._base_env.sim.model.light_dir.shape)

        # Randomize light specular
        # self._base_env.sim.model.light_specular[:] = np.random.rand(self._base_env.sim.model.nlight, 3)

        # Randomize cam FOV
        # self._base_env.sim.model.cam_fovy[:] = np.random.uniform(30, 90, size=self._base_env.sim.model.cam_fovy.shape)

        # Randomize cam pos
        # self._base_env.sim.model.cam_pos[:] += np.random.normal(0, 0.1, size=self._base_env.sim.model.cam_pos.shape)

        _, _, _, _, info = self._base_env.unwrapped.forward(update_exteroception=True)

        return self.get_cameras_pixels(info)

    def get_cameras_pixels(self, info: dict) -> torch.Tensor:
        cameras_pixels = []

        for camera_key in info["visual_dict"].keys():
            for camera_name in self.cameras:
                if (
                    camera_key.startswith("rgb:")
                    and camera_name in camera_key
                    and camera_key.endswith("2d")
                ):
                    cameras_pixels.append(info["visual_dict"][camera_key])

        return torch.from_numpy(np.array(cameras_pixels))

    def _reset(self, tensordict):
        tensordict = self._base_env._reset(tensordict)

        pixels = self.apply_visual_domain_randomization()

        tensordict.set("pixels", pixels)

        return tensordict

    def _set_seed(self, seed):
        self._base_env._set_seed(seed)


if __name__ == "__main__":
    print(f"Envs : {RoboHiveEnv.available_envs}")

    video_logger = CSVLogger(
        exp_name="visual_domain_randomization",
        log_dir="videos",
        video_format="mp4",
        video_fps=30,
    )
    recorder = VideoRecorder(logger=video_logger, tag="iteration", skip=2)

    env_name = "FetchReachRandom-v0"
    cameras = ["left_cam", "right_cam", "top_cam", "head_camera_rgb"]

    env = TransformedEnv(
        VisualDomainRandomizedEnv(
            RoboHiveEnv(
                env_name=env_name,
                from_pixels=True,
                pixels_only=True,
                from_depths=False,
                frame_skip=None,
                cameras=cameras,
            ),
            cameras=cameras,
        )
    )
    env.append_transform(recorder)
    policy = RandomPolicy(action_spec=env.action_spec)

    device = torch.device("cpu")

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=150,
        max_frames_per_traj=50,
        frames_per_batch=150,
        device=device,
        storing_device=device,
    )

    for _ in collector:
        pass

    env.transform.dump()
    env.close()
