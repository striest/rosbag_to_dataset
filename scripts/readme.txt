Step 1 : Extract all modalities except delta (/ros_talon/current_position) at dt = 0.1 
Step 2 : See if you need to incrementally add modalities if yes do so 
Step 3 : Extract delta for each traj at dt = 0.16 in a new folder
Step 4: Run Interpolate Steer script
Step 5: Run Convert to folder script