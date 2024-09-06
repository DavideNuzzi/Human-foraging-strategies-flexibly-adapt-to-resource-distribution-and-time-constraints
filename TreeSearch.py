import numpy as np
from tqdm import tqdm
from itertools import permutations


class Environment:

    def __init__(self, env_rich, battery_length, time_collect_box, time_change_room):

        # Parametri che definiscono l'environment
        self.env_rich = env_rich
        self.battery_length = battery_length

        # Parametri che dipendono dal player
        self.time_collect_box = time_collect_box
        self.time_change_room = time_change_room

        if env_rich:
            self.rooms_permutations = [[True, True, True, True]]
        else:
            self.rooms_permutations = [ [True, True, False, False], [True, False, True, False],
                                        [True, False, False, True], [False, True, True, False],
                                        [False, True, False, True], [False, False, True, True]]

        # Inizializzo il decision tree
        self.decision_tree = dict()

    # Creo un albero di decision stocastico, ciclando su tutte le possibili combinazioni dell'ambiente
    # I nodi stocastici saranno quelli che, a partire dalla stessa condizione iniziale, hanno più possibili esiti
    def create_tree(self):

        # Resetto l'albero
        self.decision_tree = dict()

        # Ciclo tra le possibili condizioni iniziali e costruisco l'albero per ognuna
        for rooms_config in self.rooms_permutations:
            self.room_rich = rooms_config
            self.create_tree_fixed_room_distribution()

    # Aggiungo al decision tree i nodi e le transizioni corrispondenti a questa configurazione
    def create_tree_fixed_room_distribution(self):

        # Creo l'albero di decisione a ambiente (distribuzione delle stanze ricche/povere) fissata
        # Parto sempre dalla stessa condizione iniziale
        state = ('U', 'U', 'U', 'U', self.battery_length, -1)
        self.search_tree(state)

    # Cerco in maniera ricorsiva tutti i possibili stati/transizioni
    def search_tree(self, state):

        # Ciclo sulle azioni possibili
        for action in range(5):

            # Chiave per il mio dizionario
            state_action = (state, action)

            # Faccio l'azione e vedo stato e ricompensa
            state_new, reward = self.do_action(state, action)

            # Se non ho mai visto questa condizione, la aggiungo
            if state_action not in self.decision_tree:
                self.decision_tree[state_action] = []
                self.decision_tree[state_action].append((state_new, reward))
            else:
                # Se l'ho già vista, controllo che sia un esito diverso, in modo da poterla poi aggiungere al mio albero come esito
                same_outcome = False
                for outcome in self.decision_tree[state_action]:
                    if (state_new, reward) == outcome:
                        same_outcome = True
                        # print(state, action, state_new, reward)
                        break
                if not same_outcome:
                    self.decision_tree[state_action].append((state_new, reward))

            if state_new != 'Terminal':
                self.search_tree(state_new)

    # Assegno un valore a ogni nodo
    def evaluate_tree(self):

        # Stato iniziale
        state = ('U', 'U', 'U', 'U', self.battery_length, -1)

        # Inizializzo il dizionario con il valore dei nodi
        # E per comodità anche uno con le coppie stato/azione
        self.state_action_values = dict()

        # Inizio la ricerca
        self.evaluate_node(state)

    # Assegno un valore a un nodo specifico
    def evaluate_node(self, state):
        
        if state in self.state_action_values:
            return np.max(self.state_action_values[state])
        
        action_values = np.zeros(5)

        # Ciclo su tutte le azioni
        for action in range(5):

            # Vedo cosa c'è nel decision tree
            transitions = self.decision_tree[(state, action)]

            # Ciclo su tutte le possibili transizioni. Se ce n'è una, la transizione è deterministica
            # Se ce ne sono di più, è stocastica e dovrò mediare
            state_action_value = 0

            for transition in transitions:
                state_new, reward = transition
                
                # Il valore di questa transizione dipende dal reward immediato più il valore del nuovo stato
                state_action_value += reward
                if state_new != 'Terminal':
                    state_action_value += self.evaluate_node(state_new)

            # Divido per il numero di scelte possibili in modo da fare il valore atteso nel caso stocastico
            state_action_value /= len(transitions)
            
            action_values[action] = state_action_value

        state_value = np.max(action_values)
        self.state_action_values[state] = action_values

        return state_value

    def get_statistics(self, samples=100000):

        coins = np.zeros(samples)
        boxes = np.zeros(samples)
        boxes_first_area = np.zeros(samples)
        visited_areas = np.zeros(samples)
        first_area_poor = np.zeros(samples, dtype=bool)

        for i in range(samples):
            results = self.simulate_episode()
            coins[i] = results[0]
            boxes[i] = results[1]
            boxes_first_area[i] = results[2]
            visited_areas[i] = results[3]
            first_area_poor[i] = results[4]

        return coins, boxes, boxes_first_area, visited_areas, first_area_poor

    def simulate_episode(self, verbose=False):

        # Inizializzo una configurazione a caso e il solito stato iniziale
        self.room_rich = self.rooms_permutations[np.random.randint(0, len(self.rooms_permutations))]
        state = ('U', 'U', 'U', 'U', self.battery_length, -1)

        # Parametri da collezionare
        coins = 0
        boxes = 0
        first_area_exit = False
        first_area_poor = False

        failsafe = 0

        while True:
            if verbose: print(f'Sono nello stato {state}')

            # Vedo i valori delle azioni
            action_values = self.state_action_values[state]

            action_values_str = [a if a >= 0 else 'Proibita' for a in action_values]
            if verbose: print(f'Valori delle azioni: {action_values_str}')

            # Scelgo l'azione migliore (in caso di pareggio, verrà scelta la prima)
            best_actions = action_values == np.max(action_values)
            best_actions_num = np.sum(best_actions)

            if best_actions_num > 1:
                if verbose: print(f'Ci sono {best_actions_num} azioni con valore identico, quindi scelgo a caso.')
                # best_actions_inds = np.where(best_actions)[0][-1]
                # action = np.random.choice(best_actions_inds)
                action = np.where(best_actions)[0][-1]
            else:
                action = np.argmax(action_values)

            if verbose: print(f'Azione scelta = {action} con valore {np.max(action_values)}')

            # Ho un solo esito perché l'ambiente è stato fissato
            state_new, reward = self.do_action(state, action)

            # Colleziono le informazioni che mi interessano
            coins += reward
            if state_new != 'Terminal': boxes += 1
            if action != 4 and state[5] != -1 and not first_area_exit:
                boxes_first_area = boxes - 1
                first_area_exit = True
            if state[5] == -1 and coins == 5: first_area_poor = True

            if verbose: print(f'Collezionate {reward} monete')

            if state_new == 'Terminal':
                if verbose: print('Scaduto il tempo, trial finito.')
                break
            else:
                state = state_new

            failsafe +=1
            if failsafe > 100:
                print('Errore, episodio non termina')

        # Il numero di aree è quello nello stato immediatamente precedente al finale (che sarà sempre terminal)
        env_state = state[0:4]
        visited_areas = np.sum([1 for i in range(4) if env_state[i] != 'U'])
        if verbose: print(f'Esplorate {visited_areas} aree. Collezionate in totale {coins} monete e {boxes} casse. Nella prima area prese {boxes_first_area} casse.')

        return coins, boxes, boxes_first_area, visited_areas, first_area_poor


    # Dato stato/azione, ottengo il nuovo stato e il reward, a ambiente fissato
    def do_action(self, state, action):

        env_state = list(state[0:4])
        time = state[4]
        player_room = state[5]

        # Vado in una stanza
        if action < 4:
            
            # Se già ero nella stanza, non faccio nulla e ottengo un disincentivo
            if player_room == action:
                reward = -1e10
                time -= 100

            # Se invece ero nella stanza iniziale vado lì e pago solo il costo di trovare la prima cassa
            elif player_room == -1:
                player_room = action

                if self.room_rich[player_room]:
                    coins = 8
                else:
                    coins = 5

                env_state[player_room] = coins-1
                reward = coins
                time -= self.time_collect_box

            # Se infine ero in un'altra stanza, vado lì e pago il costo completo. Osservo però che non posso farlo
            # se è vuota. In tal caso rimango dove stavo e ho un disincentivo
            else:

                # Provo ad andare in una stanza vuota
                if env_state[action] == "E":
                    reward = -1e10
                    time -= 100

                # Vado in una stanza ignota
                elif env_state[action] == "U":

                    player_room = action

                    if self.room_rich[player_room]:
                        coins = 8
                    else:
                        coins = 5

                    env_state[player_room] = coins-1
                    reward = coins
                    time -= self.time_change_room

                # Vado in una stanza già visitata
                else:

                    player_room = action

                    coins = env_state[player_room]
                    reward = coins
                    env_state[player_room] = coins - 1
                    time -= self.time_change_room

                    # Se la stanza ha finito le monete, cambio lo stato
                    if (self.room_rich[player_room]) and env_state[player_room] == 3:
                        env_state[player_room] = "E"
                    if (not self.room_rich[player_room]) and env_state[player_room] == 0:
                        env_state[player_room] = "E"

        # Azione di collezionare una moneta nella stanza corrente
        else:

            # Se sono nella prima stanza, devo disincentivare
            # Se la stanza è vuota, devo disincentivare
            if env_state[player_room] == "E" or player_room == -1:
                reward = -1e10
                time -= 100

            else:
                coins = env_state[player_room]
                reward = coins
                env_state[player_room] = coins - 1
                time -= self.time_collect_box

                # Se la stanza ha finito le monete, cambio lo stato
                if (self.room_rich[player_room]) and (env_state[player_room] == 3):
                    env_state[player_room] = "E"
                if (not self.room_rich[player_room]) and (env_state[player_room] == 0):
                    env_state[player_room] = "E"

        # Arrotondo i tempi a una cifra decimale
        time = round(time * 10)/10

        # Se il tempo è finito, vuol dire che questa azione mi porta allo stato terminale,
        # cosa che viene gestita automaticamente da "create_state".
        # E comunque i Q-value dello stato terminale non verranno mai aggiornati,
        # quindi non devo fare nessun controllo particolare
        if time <= 0:
            if reward > 0: reward = 0
            time = 0

        # Creo il nuovo stato
        if time == 0:
            state_new = 'Terminal'
        else:
            state_new = (*env_state, time, player_room)

        return state_new, reward




