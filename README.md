# Prediction of Configuration Space of Dual-Arm Robot via Convolutional Encoder-Decoders

## Contributions 
- First to apply a CNN to predict robotic configuration spaces from robotic worksapces. 
- Our model learns highly transferable features between robotic workspaces allowing the model to quickly adapt to translation, rotation, and removal of the obstacles from the workspaces.
- The time taken for our prediction of the configuration space image is independent of the number of number and shape of obstacles  
- Curate 4 datasets for the study of of configuration space images from robotic workspace images.

## Paper: Work in Progress. Will submit to IROS 2023.


## Model Architecture
<img width="787" alt="image" src="https://user-images.githubusercontent.com/24688175/216159037-a7e124e1-ea0a-41dd-8689-4661c851bde5.png">

## Dual Arm Robotic Workspace and Config Space
<img width="400" alt="image" src="https://user-images.githubusercontent.com/24688175/216165596-31ec98a1-89c1-4614-880b-531c32436d9c.png"><img width="500" alt="image" src="https://user-images.githubusercontent.com/24688175/216165557-27140506-bbf3-4a0f-bbcf-9bece0331d1c.png">

## (Workspace, Ground truth config-space, Predicted config space) Triples
<img width="590" alt="image" src="https://user-images.githubusercontent.com/24688175/216158430-7feb9f57-c633-4077-ba9a-192526a1fb50.png">
<img width="590" alt="image" src="https://user-images.githubusercontent.com/24688175/216158799-f153abd0-b98d-4fba-b11d-212f3dc98038.png">
<img width="590" alt="image" src="https://user-images.githubusercontent.com/24688175/216159585-00eea941-b247-4771-8726-e307ab26baef.png">

## Results
<img width="795" alt="image" src="https://user-images.githubusercontent.com/24688175/216159221-b7cf0521-69cc-4a3d-adb4-baf9b2ea343b.png">
