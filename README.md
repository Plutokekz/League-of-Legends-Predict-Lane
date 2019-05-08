# Predicting the lanes for League of Legends Participants with a CNN
## Setup
 Download the Git repository and install the requirements
 ````bash
pip install -r requirements.txt
````

## How to use
#### Training the CNN
I really recommend the use tensorflow-gpu for the training or it will take some time to train it
To run the training open the train.py file in you favorite python editor. You may change the name:
````python
if __name__ == "__main__":
    run('The Name')
````
You can also look up the Comment i tried to explain everything
If you want to see the graphs from you model while its training you can run in the console. When you are in the
League-of-Legends-Predict-Lane's folder (--logdir="Path to logs")
````bash
tensorboard --logdir=logs
````
#### Predicting Lanes
There is all ready a pre trained model in the models folder which you can use.
To predict something you import the predict function from the predict.py file and run it with a Summoner Name.
The Summoner have to be in a game to call the spectator api.
````python
from predict import predict
prediction = predict('SaItySurprise')
print(prediction)
{
 'blue':
    {
    'Lux': {'Bottom': 92.0, 'Middle': 7.4, 'Top': 0.73, 'Jungle': 0.34}, 
    'Ashe': {'Bottom': 92.0, 'Middle': 7.2, 'Top': 0.71, 'Jungle': 0.34}, 
    'Zed': {'Middle': 90.0, 'Top': 7.0, 'Bottom': 2.7, 'Jungle': 0.72}, 
    'Urgot': {'Top': 83.0, 'Jungle': 8.6, 'Middle': 6.4, 'Bottom': 1.5}, 
    'Evelynn': {'Jungle': 99.0, 'Bottom': 0.29,'Top': 0.21, 'Middle': 0.031}
    }, 
 'red': 
    {
    'Illaoi': {'Top': 83.0, 'Jungle': 8.0, 'Middle': 7.4, 'Bottom': 1.8}, 
    'Ekko': {'Middle': 91.0, 'Top': 6.7, 'Bottom': 2.0, 'Jungle': 0.64}, 
    'Soraka': {'Bottom': 92.0, 'Middle': 7.2, 'Top': 0.71, 'Jungle': 0.34}, 
    'Kaisa': {'Bottom': 92.0, 'Middle': 7.2, 'Top': 0.71, 'Jungle': 0.34}, 
    'Trundle': {'Jungle': 99.0, 'Bottom': 0.29, 'Top': 0.21, 'Middle': 0.031}
    }
}
````
If xou want to know mor about the prediction feel free to read the Comments in the prediction.py file. 
I tried to explain everything.
If there a Questions or  any suggestions feel free to Label issues and make a pull requests