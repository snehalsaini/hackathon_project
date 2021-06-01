# hackathon_project
## Approach
The project includes a data-preprocessing file which puts all the pictures in the dataset in one single folder and then produces a .csv file to indicate unqiue labels for each person's images.
The final_model.ipynb contains the pixel normalization, training and testing image function altogether. 
## Model Architecture 
### (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
### (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
### (2): ReLU(inplace=True)
### (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
### (4): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
### (5): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
### (6): ReLU(inplace=True)
### (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
### (0): Linear(in_features=196, out_features=23, bias=True)

## Steps to Execute
#### Download all the files
#### Download the combined images dataset from the link : 
#### Upload the train.csv file and dataset_images folder on googledrive
#### Run the final_model.ipynb file on Google Colab

## Some Relevant Information
The CNN training generates 25 model files and stores parameters and weights accordingly at every iteration. The final 25th model file i.e. model24 (as starting from epoch number 0) has been provided here and its weights can be used to test the model on hidden dataset. 

## Scope of Improvements
However, due to a pixel normalization in some images the model is currently trained using 100 images in the dataset and is accurately finding the match according to that. But for 4000 image, the model could still be extended, with a little modifications in image pixels pre-processing. The index error arising due to pixel normalization, is creating a gap in input features and output labels extracted which could be improved to train for all the 4420 images of the dataset.
Another approach to solve this problem could be, by using pre-trained models to extract facial features first and then train the model on those weights by using the concept of tranfer learning. But, since the problem statement was to design the full model from scratch, I tried to avoid that method. 


 
