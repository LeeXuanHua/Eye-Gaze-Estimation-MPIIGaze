# Eye-Gaze-Estimation (PyTorch Implementation of MPIIGaze and MPIIFaceGaze)
Designed to track and analyze eye movements in real-time, by leveraging computer vision and gaze estimation algorithms based on the MPIIGaze and MPIIFaceGaze papers to determine the gaze direction.

Note that this is an implementation of [existing project](https://github.com/hysts/pytorch_mpiigaze).

## Download Dataset
```bash
# Download MPIIGaze - dataset focus on cropped eye images
bash scripts/download_mpiigaze_dataset.sh
python tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/

# Download MPIIFaceGaze - dataset focus on full face images
bash scripts/download_mpiifacegaze_dataset.sh
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/
```


### Demo
To run the gaze estimation from a webcam:

1. Download the dlib pretrained model for landmark detection

    ```bash
    bash scripts/download_dlib_model.sh
    ```

2. Calibrate the camera for better accuracy (optional)

    Save  calibration result in the same format as the sample file [`data/calib/sample_params.yaml`](data/calib/sample_params.yaml)

3. Run demo

    Specify the model path and the path of the camera calibration results in the configuration file as in [`configs/demo_mpiigaze_resnet.yaml`](configs/demo_mpiigaze_resnet.yaml)

    ```bash
    python demo.py --config configs/demo_mpiigaze_resnet.yaml
    ```


### Training and Evaluation
By running the following code, you can train a model using all the data except the person with ID 0, and run test on that person.

```bash
python train.py --config configs/mpiigaze/lenet_train.yaml
python evaluate.py --config configs/mpiigaze/lenet_eval.yaml
```

Run all training and evaluation for LeNet and ResNet-8 with default parameters, using [`scripts/run_all_mpiigaze_lenet.sh`](scripts/run_all_mpiigaze_lenet.sh) and [`scripts/run_all_mpiigaze_resnet_preact.sh`](scripts/run_all_mpiigaze_resnet_preact.sh).



## Structure

- `structure` - Specifies model parameters in YAML file e.g. [`configs/mpiigaze/lenet_train.yaml`](configs/mpiigaze/lenet_train.yaml)
- `data` - camera calibration parameters, saved model files, and dlib files
- `dataset` - MPIIGaze and MPIIFaceGaze dataset downloaded earlier
- `experiments` - logs from model train & evaluation
- `gaze_estimation` - all the code data, with some notable folders:
    - `gaze_estimator` - base classes for gaze estimation specific tasks (e.g. head post estimation, face modelling)
    - `models` - neural network model definitions
- `others` - other custom code 
- `scripts` - bash scripts to automate downloading of data, executing model training & evaluation using all data, as well as reading the [mean test angle error and training time](./tools/calculate_training_time_and_errors.py)
- `tools` - tools for data preprocessing (used for [data download section](#download-dataset))


## Results

### MPIIGaze

<b> Summary</b> 

| Model           | Mean Test Angle Error [degree] | Training Time |
|:----------------|:------------------------------:|--------------:|
| LeNet           |              6.32              | 4.593 s/epoch |
| ResNet-preact-8 |              5.73              | 9.265 s/epoch |


<b> Specific Details </b> 
| Model           | Test Angle Error [degree] | Training Time [s/epoch] |
|:----------------|:-------------------------:|--------------:|
| LeNet           | ID 0: 4.201235771179199 <br> ID 1: 5.857380390167236 <br> ID 2: 6.206435680389404 <br> ID 3: 6.56096887588501 <br> ID 4: 5.819915294647217 <br> ID 5: 6.3475799560546875 <br> ID 6: 5.868472099304199 <br> ID 7: 7.9688591957092285 <br> ID 8: 6.652606010437012 <br> ID 9: 7.961040496826172 <br> ID 10: 6.87950325012207 <br> ID 11: 5.874270915985107 <br> ID 12: 6.095603942871094 <br> ID 13: 6.640603065490723 <br> ID 14: 5.9826836585998535 <br> <b> Mean: 6.327810573577881 </b> <br>|  ID 0: 4.50 <br> ID 1: 4.50 <br> ID 2: 4.70 <br> ID 3: 4.60 <br> ID 4: 4.60 <br> ID 5: 4.60 <br> ID 6: 4.60 <br> ID 7: 4.60 <br> ID 8: 4.60 <br> ID 9: 4.60 <br> ID 10: 4.60 <br> ID 11: 4.60 <br> ID 12: 4.60 <br> ID 13: 4.60 <br> ID 14: 4.60 <br> <b> Mean: 4.59 </b> <br>|
| ResNet-preact-8 | ID 0: 4.036035537719727 <br> ID 1: 4.759617805480957 <br> ID 2: 6.224761486053467 <br> ID 3: 5.100761413574219 <br> ID 4: 6.021407604217529 <br> ID 5: 6.4100799560546875 <br> ID 6: 5.680694580078125 <br> ID 7: 6.351699352264404 <br> ID 8: 6.108707904815674 <br> ID 9: 5.91240930557251 <br> ID 10: 5.642880916595459 <br> ID 11: 6.248713970184326 <br> ID 12: 5.415345668792725 <br> ID 13: 6.231785774230957 <br> ID 14: 5.865034580230713 <br> <b> Mean: 5.733995723724365 </b> <br>|  ID 0: 8.725 <br> ID 1: 8.875 <br> ID 2: 8.775 <br> ID 3: 9.000 <br> ID 4: 9.225 <br> ID 5: 9.425 <br> ID 6: 9.400 <br> ID 7: 9.150 <br> ID 8: 9.300 <br> ID 9: 9.500 <br> ID 10: 9.275 <br> ID 11: 9.600 <br> ID 12: 9.500 <br> ID 13: 9.675 <br> ID 14: 9.550 <br> <b> Mean: 9.265 </b> <br>|

The training time is the value when using RTX3090.

There is a slight difference with the training done by the sample code, as we did not set `cudnn benchmark = False`, `cudnn deterministic = True`

### MPIIFaceGaze

<b> Summary</b> 

| Model     | Mean Test Angle Error [degree] | Training Time |
|:----------|:------------------------------:|--------------:|
| AlexNet   |              5.09              |  131 s/epoch  |
| ResNet-14 |              4.82              |   87 s/epoch  |

<b> Specific Details </b> 
| Model           | Test Angle Error [degree] | Training Time [s/epoch] |
|:----------------|:-------------------------:|--------------:|
| Alexnet           | ID 0: 2.5926711559295654 <br> ID 1: 5.880463123321533 <br> ID 2: 5.201772212982178 <br> ID 3: 5.638153553009033 <br> ID 4: 4.7275776863098145 <br> ID 5: 4.894957542419434 <br> ID 6: 3.4403390884399414 <br> ID 7: 5.164022922515869 <br> ID 8: 5.124176979064941 <br> ID 9: 5.189162254333496 <br> ID 10: 5.192043781280518 <br> ID 11: 5.762932300567627 <br> ID 12: 5.050137519836426 <br> ID 13: 5.1160736083984375 <br> ID 14: 7.457568168640137 <br> <b> Mean: 5.09547012646993 </b> <br>|  ID 0: 132 <br> ID 1: 131 <br> ID 2: 131 <br> ID 3: 131 <br> ID 4: 131 <br> ID 5: 131 <br> ID 6: 131 <br> ID 7: 131 <br> ID 8: 130 <br> ID 9: 131 <br> ID 10: 131 <br> ID 11: 131 <br> ID 12: 132 <br> ID 13: 131 <br> ID 14: 131 <br> <b> Mean: 131 </b> <br>|
| ResNet-14 | ID 0: 2.1106247901916504 <br> ID 1: 5.465665817260742 <br> ID 2: 5.1382951736450195 <br> ID 3: 6.074862003326416 <br> ID 4: 3.6258327960968018 <br> ID 5: 4.259786605834961 <br> ID 6: 3.0125956535339355 <br> ID 7: 4.566280841827393 <br> ID 8: 4.682868480682373 <br> ID 9: 5.128609657287598 <br> ID 10: 5.466821670532227 <br> ID 11: 6.558365345001221 <br> ID 12: 4.866367340087891 <br> ID 13: 4.26247501373291 <br> ID 14: 7.144346237182617 <br> <b> Mean: 4.82425316174825 </b> <br>|  ID 0: 77 <br> ID 1: 77 <br> ID 2: 77 <br> ID 3: 79 <br> ID 4: 110 <br> ID 5: 112 <br> ID 6: 112 <br> ID 7: 110 <br> ID 8: 111 <br> ID 9: 112 <br> ID 10: 66 <br> ID 11: 66 <br> ID 12: 66 <br> ID 13: 66 <br> ID 14: 69 <br> <b> Mean: 87 </b> <br>|

The training time is the value when using RTX3090.

There is a slight difference with the training done by the sample code, as we did not set `cudnn benchmark = False`, `cudnn deterministic = True`