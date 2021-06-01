# hackathon_project
## Approach
The project includes a data-preprocessing file which puts all the pictures in the dataset in one single folder and then produces a .csv file to indicate unqiue labels for each person's images.
The final_model.ipynb contains the pixel normalization, training and testing image function altogether. 
##Model Architecture
(
    (0): Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (0): Linear(in_features=196, out_features=23, bias=True)
  )
 
