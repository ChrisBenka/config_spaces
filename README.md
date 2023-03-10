# Direct Robot Configuration Space Construction using Convolutional Encoder-Decoders 
### SUBMITED TO IROS 2023

## Contributions 
- We are the first to construct C-spaces directly from robotic workspaces using a convolutional encoder-decoder. The model achieves a$97.5\% F1-score for the prediction of configurations that belong to C-Free and C-clsn and limits undetected collisions to less than 2.5\%.
Our model learns highly transferable features between robotic workspaces involving new transformations on obstacles. After training on translation of obstacles, the model adapts to the removal and rotation of obstacles with little to no fine-tuning.

## Paper: 

## Workspace
<img width="495" alt="image" src="https://user-images.githubusercontent.com/24688175/224188874-4f1c0aa9-0a67-427c-b17a-4ea604d5e78c.png">

## Model Architecture
<img width="787" alt="image" src="https://user-images.githubusercontent.com/24688175/216159037-a7e124e1-ea0a-41dd-8689-4661c851bde5.png">



## Main results
### Tables
<img width="937" alt="image" src="https://user-images.githubusercontent.com/24688175/224188799-2d544603-4594-4bf8-94f1-32df342b9cfb.png">
### Images
<img width="431" alt="image" src="https://user-images.githubusercontent.com/24688175/224188051-48edac98-edd2-4601-8313-1930867581f8.png">

### Zero-Shot Images
<img width="448" alt="image" src="https://user-images.githubusercontent.com/24688175/224188470-d8f41828-b1d1-4eee-abc4-4b47eb21f6a8.png">
