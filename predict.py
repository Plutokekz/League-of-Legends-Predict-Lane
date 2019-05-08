import cassiopeia as cass
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
from private import key


def setup(model_name='test4_conv1D_153_one_dense_153_1557335309_1557335316'):
    # Configuring the tenserflow/keras session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True      # to log device placement (on which device the operation ran)
                                            # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)

    # Setup the MinMaxScaler to scale the input Data right
    min_max_scaler = MinMaxScaler()
    with open('data/X.pickle', 'rb') as file:
        X = pickle.load(file)
    min_max = min_max_scaler.fit(X)

    # Loading the model in
    model = load_model(f'models/{model_name}')
    return model, min_max


def prepare(summoner_name):
    # Trying the get the running game by the summoner Name
    cass.set_riot_api_key(key)
    cass.set_default_region('EUW')
    summoner = cass.get_summoner(name=summoner_name, region='EUW')
    try:
        current_game = summoner.current_match()
    except Exception as e:
        print(str(e))
        return None
    if current_game.mode == cass.data.GameMode.aram:
        print('there are no lanes in Aram')
        return None
    # Preparing the participants per team
    for team in current_game.teams:
        yield prepare_team(team), team


def prepare_team(team):
    # getting the data in the right format, one input array for the network hast to look like:
    # participant = champId, spell1Id, spell2Id
    # [ 1, participant1, participant2, participant3, participant4, participant5]
    # the 1 shows marks the champion which lanes get predicted, A hole team looks like this:
    # [[ 1, participant1, participant2, participant3, participant4, participant5],
    # [ participant1, 1, participant2, participant3, participant4, participant5],
    # [ participant1, participant2, 1, participant3, participant4, participant5],
    # [ participant1, participant2, participant3, 1, participant4, participant5],
    # [ participant1, participant2, participant3, participant4, 1, participant5]]

    participants, champions = [], []
    for i in range(0, 5):
        prepared_participant = []
        for index, participant in enumerate(team.participants):
            if index == i:
                # Get the champions Names the map them to there lanes at the End
                champions.append(participant.champion.key)
                prepared_participant.append(1)
            prepared_participant.append(participant.champion.id)
            prepared_participant.append(participant.summoner_spell_d.id)
            prepared_participant.append(participant.summoner_spell_f.id)
        participants.append(prepared_participant)
    return participants, champions


def predict(summoner_name):
    model, min_max = setup()
    prepared = prepare(summoner_name)
    if prepared:
        teams = {}
        for data, team in prepared:
            team_data, team_champions = data

            # Scaling Data to range 0, 1
            X = min_max.transform(team_data)
            # reshaping the Data for the Convolutional Network input
            X = X.reshape(5, 16, 1)
            predictions = model.predict(X)

            # Converting prediction to something readable
            prediction_dict = {}
            for champion, predicted_lanes in zip(team_champions, predictions):
                predicted_lanes = [float(f'{value*100:.2}') for value in predicted_lanes]

                # Splitting up the prediction for each lane
                lanes = {'Top': predicted_lanes[0], 'Middle': predicted_lanes[1], 'Bottom': predicted_lanes[2],
                     'Jungle': predicted_lanes[3]}

                # Sorting lanes for there probability/confidence that there are right
                lanes = sorted(lanes.items(), key=lambda kv: kv[1], reverse=True)
                lanes = dict(lanes)
                prediction_dict[champion] = lanes
            teams[team.side.name] = prediction_dict
        return teams

