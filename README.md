# RoboHiveTorchRL-Prototype
Small prototype to show RoboHive usage with TorchRL.

# Setup
1. ```shell
    micromamba create -n robohive -f environment.yaml -y
   ```
2. ```shell
    micromamba activate robohive
    ```
3. ```shell
    robohive_init
   ```  

# Explore Random Environments
1. ```shell
    python explore_robohive_envs.py
    ```
2. Look under the newly created `videos/` for videos of the explored envs. The script explores a few envs by default. 

## Examples  
<video controls>
  <source src="videos/FrankaPushRandom-v0/videos/iteration_0.mp4" type="video/mp4">
</video>
<video controls>
  <source src="videos/FK1_LdoorOpenRandom-v4/videos/iteration_0.mp4" type="video/mp4">
</video>
<video controls>
  <source src="videos/DKittyStandRandom-v0/videos/iteration_0.mp4" type="video/mp4">
</video>
