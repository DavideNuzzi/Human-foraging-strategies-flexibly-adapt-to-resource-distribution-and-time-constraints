import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
from TreeSearch import Environment
from matplotlib import pyplot as plt
from colorama import Fore, Style
from colorama import init as colorama_init
from copy import deepcopy
from multiprocessing import Pool


def worker(args):
    find_optimal_agent(*args)

def find_optimal_agent(subject_ind, box_find_time, room_change_time):
   
    start_time = time.time()
    
    # Dove metterò tutti i risultati
    results = dict()
    models = dict()
    variables = ['Coins','Boxes','BoxesFirstArea','VisitedAreas']
 
    for env_type in ['Poor','Rich']:
        for battery_length in [25,50]:
            
            if env_type == 'Poor': is_rich = False
            else: is_rich = True
            if battery_length == 25: battery_type = 'Short'
            else: battery_type = 'Long'

            if type(box_find_time) is list:
                if battery_length == 25: 
                    time_collect_box = box_find_time[0]
                    time_change_room = room_change_time[0]
                else: 
                    time_collect_box = box_find_time[1]
                    time_change_room = room_change_time[1]
            else:
                time_collect_box = box_find_time
                time_change_room = room_change_time

            # Arrotondo i tempi a 1 cifra decimale, perché sono comunque già molto approssimativi e così rendo più veloci gli algoritmi
            time_collect_box = np.round(time_collect_box * 10)/10
            time_change_room = np.round(time_change_room * 10)/10
            
            # Creo l'ambiente
            env = Environment(env_rich=is_rich, battery_length=battery_length, time_collect_box=time_collect_box, time_change_room=time_change_room)

            # Creo l'albero di decisione
            env.create_tree()
            env.evaluate_tree()

            # Risultati
            coins, boxes, boxes_first_area, visited_areas, first_area_poor = env.get_statistics()

            # Salvo le variabili di interesse
            for v in variables:
                if v == 'Coins': x = np.mean(coins)
                if v == 'Boxes': x = np.mean(boxes)
                if v == 'BoxesFirstArea': x = np.mean(boxes_first_area)
                if v == 'VisitedAreas': x = np.mean(visited_areas)

                name = v + '_' + battery_type + '_' + env_type
                results[name] = x

                # Se sono nel caso poor, devo anche separare
                if env_type == 'Poor':
                    if v == 'Coins': 
                        x1 = np.mean(coins[first_area_poor == True])
                        x2 = np.mean(coins[first_area_poor == False])
                    if v == 'Boxes':
                        x1 = np.mean(boxes[first_area_poor == True])
                        x2 = np.mean(boxes[first_area_poor == False])
                    if v == 'BoxesFirstArea':
                        x1 = np.mean(boxes_first_area[first_area_poor == True])
                        x2 = np.mean(boxes_first_area[first_area_poor == False])
                    if v == 'VisitedAreas':
                        x1 = np.mean(visited_areas[first_area_poor == True])
                        x2 = np.mean(visited_areas[first_area_poor == False])

                    name_1 = name + '_FirstPoor'
                    name_2 = name + '_FirstRich'

                    results[name_1] = x1
                    results[name_2] = x2                

            # Salvo il modello
            name = battery_type + '_' + env_type
            models[name] = deepcopy(env)


    # Salvo i risultati
    with open(f'ResultsAgent_Subject_{subject_ind}_NoSplit.pkl','wb') as f:
        pkl.dump([results, models], f)

    # Print conclusivo
    print('\n\n')
    print(f'Soggetto {subject_ind}\n')
    print(f'Tempo tra box = {Fore.GREEN}{box_find_time}s{Style.RESET_ALL}\t\tTempo per cambiare stanza = {Fore.GREEN}{room_change_time}s{Style.RESET_ALL}\n')

    print('\t\t\t\t\tLong Battery\t\t\t\t\tShortBattery\t')
    print('\t\t\tRich\tPoor\tFirstRich\tFirstPoor\t|\tRich\tPoor\tFirstRich\tFirstPoor')
    for v in variables:
        line = f'{v}:\t'
        if v == 'Coins' or v == 'Boxes': line += '\t'
        for battery_type in ['Long','Short']:
            for env_type in ['Rich','Poor']:
                name = v + '_' + battery_type + '_' + env_type
                x = results[name]
                if x < 0: x = -1
                line += f'\t{x:.2f}'

                if env_type == 'Poor':
                    name_1 = name + '_FirstRich'
                    name_2 = name + '_FirstPoor'
                    x1 = results[name_1]
                    x2 = results[name_2]
                    if x1 < 0: x1 = -1
                    if x2 < 0: x2 = -1
                    line += f'\t{x1:.2f}\t\t{x2:.2f}\t'

            line += '\t|'

        print(line)

    end_time = time.time()
    print(f'Tempo impiegato {end_time - start_time}')
    print('\n\n')


def main():

    # Carico i dati e vedo i tempi
    data = pd.read_csv('results_tempi.csv') 

    box_find_times = np.zeros(34)
    room_change_and_first_box_times = np.zeros(34)
    box_find_times_short = np.zeros(34)
    box_find_times_long = np.zeros(34)

    room_change_and_first_box_times_short = np.zeros(34)
    room_change_and_first_box_times_long = np.zeros(34)


    for i in range(34):   
        data_subject = data[(data["SubjectCode"] == (i + 1))]

        b_times = data_subject['TempiTraCasse']
        b_times = [float(x) for s in b_times for x in s.replace('[','').replace(']','').split(',')]
        box_find_times[i] = np.nanmean(b_times)  

        b_times = data_subject[data_subject['LongBattery'] == True]['TempiTraCasse']
        b_times = [float(x) for s in b_times for x in s.replace('[','').replace(']','').split(',')]
        box_find_times_long[i] = np.nanmean(b_times)  

        b_times = data_subject[data_subject['LongBattery'] == False]['TempiTraCasse']
        b_times = [float(x) for s in b_times for x in s.replace('[','').replace(']','').split(',')]
        box_find_times_short[i] = np.nanmean(b_times)  

        r_times = data_subject['TempiUltimaPrimaNuovaArea']
        r_times = [float(x) for s in r_times for x in s.replace('[','').replace(']','').split(',') if len(s) > 2]
        room_change_and_first_box_times[i] = np.nanmean(r_times)  

        r_times = data_subject[data_subject['LongBattery'] == True]['TempiUltimaPrimaNuovaArea']
        r_times = [float(x) for s in r_times for x in s.replace('[','').replace(']','').split(',') if len(s) > 2]
        room_change_and_first_box_times_long[i] = np.nanmean(r_times)  

        r_times = data_subject[data_subject['LongBattery'] == False]['TempiUltimaPrimaNuovaArea']
        r_times = [float(x) for s in r_times for x in s.replace('[','').replace(']','').split(',') if len(s) > 2]
        room_change_and_first_box_times_short[i] = np.nanmean(r_times)  


    inputs = [(i, box_find_times[i], room_change_and_first_box_times[i]) for i in range(34)]
 



if __name__ == "__main__":
    main()