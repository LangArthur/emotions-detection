# Emotion-detection

The program detects different emotions of a person from video stream/images.

This project has an educational purpose.

## Installation

To install all dependancies, we recommande to use pip.

```
pip install -r requirements.txt
```

## Run the program

### Default mode (CNNv2 model for detection)
python main.py
### Changing the model for detection
python main.py nameOfTheModel

## Model availables

In this project, three architectures were implemented:

1. ModelInception - inspired from article [1].
2. SimpleCNN - model base on [this implementation](https://github.com/MinG822/ferpredict3).
3. CNNv2 - model base on [this implementation](https://github.com/atulapra/Emotion-detection).

The best performances were reached using CNNv2 model trained on [FER-2013 dataset](https://www.kaggle.com/msambare/fer2013). The program uses this model by defalt when running, but the model can be changed by specifying the model name on command line (available models: 'Inception', 'SimpleCNN', 'CNNv2')

## References:

1. A. Mollahosseini, D. Chan and M. H. Mahoor, "Going deeper in facial expression recognition using deep neural networks," 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), 2016, pp. 1-10, doi: 10.1109/WACV.2016.7477450.
2. Mollahosseini, Ali & Chan, David & Mahoor, Mohammad. (2016). Going deeper in facial expression recognition using deep neural networks. 1-10. 10.1109/WACV.2016.7477450. 
