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

## Example 
<video width="320" height="240" controls>
  <source src="https://github.com/AlexandreBrown/RoboHiveTorchRL-Prototype/blob/main/videos/FrankaPushRandom-v0/videos/iteration_0.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>