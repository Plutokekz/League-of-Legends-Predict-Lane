import pickle

import cassiopeia as cass
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


class Lane:

    def __init__(self, api_key, **kwargs):
        """
        :param api_key:
        :param kwargs:
        """
        if 'model_name' in kwargs.keys():
            model_name = kwargs['model_name']
        else:
            model_name = 'tf2.04_conv1D_153_one_dense_153_1578830710'  # 'test4_conv1D_153_one_dense_153_1557335309_1557335316'
        self.key = api_key
        self.model, self.min_max = self._setup(model_name)

    def _prepare(self, id_or_name):
        """
        Get's the Teams from a live game by Summoner Name or from an old Match by Match ID
        :param id_or_name: either a Summoner Name or a Match ID
        :return: iterator of the prepared teams; if not exits None
        """
        # Setting the Api key for Cassiopeia and the default region
        cass.set_riot_api_key(self.key)
        cass.set_default_region('EUW')
        if type(id_or_name) is str:
            summoner = cass.get_summoner(name=id_or_name, region='EUW')
            try:
                current_game = summoner.current_match()
            except Exception as e:
                print(str(e))
                return None
        else:
            try:
                current_game = cass.get_match(id_or_name, region='EUW')
            except Exception as e:
                print(str(e))
                return None
        if current_game.mode == cass.data.GameMode.aram:
            print('there are no lanes in Aram')
            return None
        # Preparing the participants per team
        for team in current_game.teams:
            yield self._prepare_team(team), team

    def _prepare_team(self, team):
        """
        Preparing a Team
        getting the data in the right format, one input array for the network hast to look like:
        participant = champId, spell1Id, spell2Id
        [ 1, participant1, participant2, participant3, participant4, participant5]
        the 1 shows marks the champion which lanes get predicted, A hole team looks like this:
        [[ 1, participant1,    participant2,    participant3,    participant4,    participant5],
        [     participant1, 1, participant2,    participant3,    participant4,    participant5],
        [     participant1,    participant2, 1, participant3,    participant4,    participant5],
        [     participant1,    participant2,    participant3, 1, participant4,    participant5],
        [     participant1,    participant2,    participant3,    participant4, 1, participant5]]
        :param team:
        :return:
        """
        # the champions list a just the names of the champions to visualize it at the end
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

    def predict(self, id_or_name):
        """
        :param id_or_name: you can insert a summoner name for predicting a live match or a matchId to analyse past matches
        :return: dict(teams)
        """
        prepared = self._prepare(id_or_name)
        if prepared:
            teams = {}
            for data, team in prepared:
                team_data, team_champions = data

                # Scaling Data to range 0, 1
                X = self.min_max.transform(team_data)
                # reshaping the Data for the Convolutional Network input
                X = X.reshape(5, 16, 1)
                predictions = self.model.predict(X)

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

    def _setup(self, model_name):
        """
        Loading the Model and the pickled championIds for min max scaling
        :param model_name:
        :return: Keras model, sklearn min_mac_sclaer
        """
        # Loading the model in
        try:
            model = load_model(f'models/{model_name}')
            # model._make_predict_function()
        except Exception as e:
            print(str(e))
            exit()
        # Setup the MinMaxScaler to scale the input Data right
        min_max_scaler = MinMaxScaler()
        with open('data/X.pickle', 'rb') as file:
            X = pickle.load(file)
        min_max = min_max_scaler.fit(X)
        return model, min_max
