
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, linregress
from matplotlib import pyplot as plt
from matplotlib.cm import viridis, jet
from matplotlib import patches
import numpy as np






def boxplot(data, test_type='mann-whitney', test_combinations=None, multiple_correction=None, custom_significant_combinations=None, show_points=False, show_connecting_lines=False, sep_multiplier=1, connecting_lines_skip=1, box_colors='w', median_colors='k', boxes_alpha=1, linewdith=1, median_linewidth=1, points_colors='gradient', significance_lines_position='up'):
    

    # Se i dati sono una matrice di numpy, li trasformo in una lista di array
    if type(data) is np.ndarray:
        data = [data[:, i] for i in range(data.shape[1])]

    # Converto per sicurezza tutti gli elementi delle liste in array di numpy 
    data = [np.array(d) for d in data]

    # Elimino eventuali nan
    data = [d[np.isnan(d) == False] for d in data]

    # Info sui dati
    boxes_num = len(data)
    samples_num = len(data[0])
    same_num = sum([True if len(data[i]) == len(data[0]) else False for i in range(boxes_num)]) == boxes_num

    # Mostro il boxplot usando la funzione di matplotlib
    bplot = plt.boxplot(data, patch_artist=True)

    # Coloro i boxplot e cambio le dimensioni delle linee
    for i, (patch, median) in enumerate(zip(bplot['boxes'], bplot['medians'])):

        if type(box_colors) is str: b_col = box_colors
        else:                       b_col = box_colors[i]
           
        if type(median_colors) is str: m_col = median_colors
        else:                       m_col = median_colors[i]
              
        patch.set_facecolor(b_col)
        median.set_color(m_col)
        patch.set_alpha(boxes_alpha)
        patch.set_linewidth(linewdith)
        median.set_linewidth(median_linewidth)

    # Mostro i punti
    if show_connecting_lines:
        show_points = True

    if show_points:
        
        if same_num:
            if type(points_colors) == str:
                if points_colors == 'random':
                    points_colors = [[np.random.rand(3) for _ in range(samples_num)]] * boxes_num

                elif points_colors == 'gradient':

                    # Ordino i sample dall'alto verso il basso nel primo box e uso quelli come riferimento
                    if connecting_lines_skip == 1:
                        y = np.array(data[0])
                        y_ind_sort = np.argsort(y)
                        points_colors = np.zeros((samples_num, 4))
                        for i in range(samples_num):
                            points_colors[y_ind_sort[i],:] = viridis(i/samples_num)
                        points_colors = [points_colors] * boxes_num
                    else:
                        points_colors = []
                        for i in range(0, boxes_num):
                            if i % connecting_lines_skip == 0:
                                y = np.array(data[i])
                                y_ind_sort = np.argsort(y)
                            p_col = np.zeros((samples_num,4))
                            
                            for j in range(samples_num):
                                p_col[y_ind_sort[j], :] = viridis(j/samples_num)
                            points_colors.append(p_col)

                elif points_colors == 'w':

                    points_colors = [np.ones((samples_num,3))] * boxes_num

            elif type(points_colors) == np.ndarray:

                points_colors = [points_colors] * boxes_num



        else:
            points_colors = [[0, 0, 0]] * boxes_num
            print('Il tipo di colore richiesto per i punti è incompatibile con il fatto che ogni box ha un numero diverso di punti')

        for i in range(boxes_num-1):

            x_values_1 = np.ones(len(data[i])) * i + 1
            x_values_2 = np.ones(len(data[i+1])) * i + 2
            y_values_1 = data[i]
            y_values_2 = data[i+1]

            # Il colore dei punti o è dato da un gradiente (dall'alto in basso)
            # Oppure è dato dall'utente (per ogni box) oppure da un gradiente
            col_1 = points_colors[i]
            col_2 = points_colors[i+1]
        
            # Punti
            plt.scatter(x_values_1, y_values_1, 10, color=col_1, zorder=10, edgecolors='k')
            plt.scatter(x_values_2, y_values_2, 10, color=col_2, zorder=10, edgecolors='k')

    # Connetto tra loro i punti dei vari boxplot, ma solo per quelli adiacenti
    if show_connecting_lines:

        if same_num:

            for i in range(0,boxes_num-1,connecting_lines_skip):
                x_values_1 = np.ones(samples_num) * i + 1
                x_values_2 = np.ones(samples_num) * i + 2
                y_values_1 = data[i]
                y_values_2 = data[i+1]

                for j in range(samples_num):
                    if y_values_2[j] > y_values_1[j]:
                        line_col = 'g'
                    else:
                        line_col = 'r'

                    plt.plot([x_values_1[j], x_values_2[j]], [y_values_1[j], y_values_2[j]], line_col, alpha=0.2, linewidth=1)
        else:
            print("E' stato richiesto di mostrare le linee che connettono i punti nei box, ma ogni box ha un numero di punti diverso")

    # Creazione combinazioni per il test
    if test_combinations == 'all':
        test_combinations = [(i, j) for i in range(boxes_num) for j in range(i+1, boxes_num)]

    # Faccio i test
    if custom_significant_combinations is None:
        significant_combinations = []

        if test_combinations is not None:
            combinations_num = len(test_combinations)

            for c in test_combinations:

                data_1 = data[c[0]]
                data_2 = data[c[1]]

                if test_type == 't-test':
                    _, p = ttest_ind(data_1, data_2, alternative='two-sided')
                if test_type == 'mann-whitney':
                    _, p = mannwhitneyu(data_1, data_2, alternative='two-sided')

                if p <= 0.05:
                    significant_combinations.append((*c, p))

        # Correzione di Bonferroni
        if multiple_correction == 'bonferroni':
            for i, c in enumerate(significant_combinations):

                ind_1, ind_2, p = c  
                p_mod = p * combinations_num

                if p_mod <= 0.05:
                    significant_combinations[i] = (ind_1, ind_2, p_mod)
                else:
                    significant_combinations.pop(i)
                    i -= 1

        # False discovery rate
        if multiple_correction == 'false-discovery':

            # Ordino la lista di tutti i p-value
            p_values = [c[2] for c in significant_combinations]
            p_sorted_inds = np.argsort(p_values)
            combinations_to_remove = []

            for i in range(len(p_sorted_inds)):
                p_ind = p_sorted_inds[i]
                ind_1, ind_2, p = significant_combinations[p_ind]
                p_mod = p * combinations_num / (i + 1)

                if p_mod <= 0.05:
                    significant_combinations[p_ind] = (ind_1, ind_2, p_mod)
                else:
                    combinations_to_remove.append(significant_combinations[p_ind] )

            for c in combinations_to_remove:
                significant_combinations.remove(c)
    else:
        significant_combinations = custom_significant_combinations

    # Plotto le differenze significative
    if len(significant_combinations) > 0:

        print(significant_combinations)
        # Altezza iniziale per le linee
        ylim = plt.ylim()
        y_shift = 0.03 * (ylim[1] - ylim[0]) * sep_multiplier

        # Riordino le combinazioni in funzione della distanza tra i boxplot che le compongono
        box_dist = [abs(c[0] - c[1]) for c in significant_combinations]
        box_order = np.argsort(box_dist)
        significant_combinations_ordered = [significant_combinations[box_order[i]] for i in range(len(significant_combinations))]

        line_heights_up = np.array([np.max(d) for d in data])
        line_heights_down = np.array([np.min(d) for d in data])

        for i, c in enumerate(significant_combinations_ordered):
            ind1, ind2, p_value = c

            if p_value < 0.001:
                asterisks = '***'
            elif p_value < 0.01:
                asterisks = '**'
            elif p_value < 0.05:
                asterisks = '*'

            # Vedo l'altezza massima registrata di tutti i boxplot tra questi due
            # compresi essi stessi
            if significance_lines_position == 'up':
                line_pos = 'up'
            elif significance_lines_position == 'down':
                line_pos = 'down'
            elif type(significance_lines_position) is dict:
                # Cerco quella corrispondente
                if (ind1, ind2) in significance_lines_position:
                    line_pos = significance_lines_position[(ind1, ind2)]
                else:
                    line_pos = 'up'
            else:
                print('Tipo di posizionamento linee significatività non supportato')

            if line_pos == 'up':
                height = np.max(line_heights_up[ind1:(ind2+1)]) + y_shift
                tips_height = height
                line_height = tips_height + y_shift / 2
                line_heights_up[ind1:(ind2+1)] = height + y_shift * 2

            if line_pos == 'down':
                height = np.min(line_heights_down[ind1:(ind2+1)]) - y_shift
                tips_height = height
                line_height = tips_height - y_shift / 2
                line_heights_down[ind1:(ind2+1)] = height - y_shift * 2

            # Draw the significance line
            plt.plot([ind1 + 1, ind1 + 1, ind2 + 1, ind2 + 1], [tips_height, line_height, line_height, tips_height], lw=1, c='k')
           
            # Draw the asterisk for significance
            if line_pos == 'up':
                plt.text((ind1 + ind2 + 2) * .5, line_height - y_shift * 0.5, asterisks, ha='center', va='bottom', color=[0.2,0.2,0.2])
            if line_pos == 'down':
                plt.text((ind1 + ind2 + 2) * .5, line_height - y_shift * 0.5, asterisks, ha='center', va='top', color=[0.2,0.2,0.2])



def get_data_for_condition(df, variable, conditions):

    result = []

    # Ciclo tra le condizioni
    for i, condition in enumerate(conditions):

        # Filtro i dati di questa condizione
        mask = None

        for var in condition:
            if mask is None:
                mask = df[var] == condition[var]
            else:
                mask = mask & (df[var] == condition[var])

        data_condition = df[mask]

        # Raggruppo per soggetto e medio su ogni variabile di interesse
        data_condition_by_subject = data_condition.groupby('Subject')

        var_mean = data_condition_by_subject[variable].mean().to_numpy()

        result.append(var_mean)

    return result





def boxplot_2x2(ax, df, variable, condition, significant_pairs, box_colors, median_colors, xlabels, ylabel, legend_labels, only_top_lines=False):

    plt.sca(ax)

    # Estraggo i dati che mi interessano
    x_cond = get_data_for_condition(df, variable, condition)

    significance_line_position = {(0,2):'down',(1,3):'down'}
    if only_top_lines is True: significance_line_position = 'up'
    
    box_colors = box_colors*2
    median_colors = median_colors*2

    # Plot
    boxplot(x_cond, custom_significant_combinations=significant_pairs, boxes_alpha=0.8, box_colors=box_colors, median_colors=median_colors, significance_lines_position = significance_line_position)
    
    # Allargo un po' il range in modo da far respirare il plot
    y_lim = plt.ylim()
    y_range = (y_lim[1] - y_lim[0])
    space = y_range * 0.05
    y_lim_new = [y_lim[0] - space, y_lim[1] + space]

    # Disegno la linea verticale come separatore e 
    plt.plot([2.5,2.5], y_lim_new,':k')
    plt.ylim(y_lim_new)

    plt.xticks([1.5,3.5], xlabels, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, linestyle=':')

    # patch_1 = patches.Patch(facecolor=box_colors[0], alpha=0.8, edgecolor='k', label = legend_labels[0])
    # patch_2 = patches.Patch(facecolor=box_colors[1], alpha=0.8, edgecolor='k',  label = legend_labels[1])
    # plt.legend(handles = [patch_1, patch_2], loc='upper right', fontsize=12)







def linear_fit_errors(x_mean, y_mean, x_std, y_std, samples=200):

    N = len(x_mean)
    slopes = np.zeros(samples)
    intercepts = np.zeros(samples)
    corrcoeff = np.zeros(samples)

    if x_std is None: x_std = 0
    if y_std is None: y_std = 0
    
    for n in range(samples):

        x = np.random.randn(N) * x_std + x_mean
        y = np.random.randn(N) * y_std + y_mean

        res = linregress(x,y)
        m = res.slope
        q = res.intercept
        r = res.rvalue

        slopes[n] = m
        intercepts[n] = q
        corrcoeff[n] = r

    return slopes, intercepts, corrcoeff



def linear_plot_errors(x_mean, y_mean, x_std=None, y_std=None, samples=1000, outliers_sigma=3, discard_outliers=False, show_outliers=True, col=[0,0,1], linecol=[1,0,0], show_errorbars=True, area_alpha=0.2):

    mask_good = None
    outlier_inds = []

    if discard_outliers:
        if x_std is not None:
            x_std_std = np.std(x_std)
            mask_good = ( np.abs( (x_std - np.mean(x_std))/x_std_std) <  outliers_sigma)
        if y_std is not None:
            y_std_std = np.std(y_std)

            if x_std is not None:
                mask_good = mask_good & ( np.abs( (y_std - np.mean(y_std))/y_std_std) <  outliers_sigma)
            else:
                mask_good = ( np.abs( (y_std - np.mean(y_std))/y_std_std) <  outliers_sigma)
        
    if mask_good is None:
        slopes, intercepts, corrcoeff = linear_fit_errors(x_mean, y_mean, x_std, y_std, samples=samples)
    else:
        x_std_good = None
        y_std_good = None
        outlier_inds = np.where(mask_good == False)[0]
        if x_std is not None: x_std_good = x_std[mask_good]
        if y_std is not None: y_std_good = y_std[mask_good]
        slopes, intercepts, corrcoeff = linear_fit_errors(x_mean[mask_good], y_mean[mask_good], x_std_good, y_std_good, samples=samples)
        
        print(f'Rimossi {len(x_mean) - np.sum(mask_good)} outliers')
    
    x_min = np.min(x_mean)
    x_max = np.max(x_mean)

    if x_std is not None:
        x_min = np.min(x_mean - x_std)
        x_max = np.max(x_mean + x_std)

    if mask_good is not None:
        x_min = np.min(x_mean[mask_good])
        x_max = np.max(x_mean[mask_good])

        if x_std is not None:
            x_min = np.min(x_mean[mask_good] - x_std[mask_good])
            x_max = np.max(x_mean[mask_good] + x_std[mask_good])

    deltax = x_max - x_min
    x_min -= deltax * 0.2
    x_max += deltax * 0.2
    x_interp = np.linspace(x_min, x_max, 60)

    y_interp_mean = np.zeros(len(x_interp))
    y_interp_min = np.zeros(len(x_interp))
    y_interp_max = np.zeros(len(x_interp))

    for i in range(len(x_interp)):
        x = x_interp[i]
        y_interp_mean[i] = np.mean(slopes) * x + np.mean(intercepts)

        y_distribution = np.zeros(samples)

        for n in range(samples):
            # slope = np.random.choice(slopes)
            # intercept = np.random.choice(intercepts)

            slope = slopes[n]
            intercept = intercepts[n]

            y = slope * x + intercept
            y_distribution[n] = y

        y_s = np.std(y_distribution)
        y_m = np.mean(y_distribution) - y_s
        y_M = np.mean(y_distribution) + y_s
        
        y_m = np.percentile(y_distribution,5)
        y_M = np.percentile(y_distribution,95)

        y_interp_min[i] = y_m
        y_interp_max[i] = y_M

    # Ora plotto
    # Dati con errore
    ecol = [col[0],col[1],col[2],0.2]
    if show_outliers:
        if show_errorbars:
            plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, marker='o', markersize=5, linestyle='none', capsize=3, ecolor=ecol)
        else:
            plt.plot(x_mean, y_mean, marker='.', markersize=8, color=col, linestyle='none')
    else:
        plt.errorbar(x_mean[mask_good], y_mean[mask_good], xerr=x_std[mask_good], yerr=y_std[mask_good], marker='o', markersize=5, linestyle='none', capsize=3, ecolor=ecol)

    # Linea
    plt.plot(x_interp, y_interp_mean, color=linecol)

    # Tutte le linee intermedie?
    # for n in range(samples):
    #     plt.plot(x_interp, x_interp * slopes[n] + intercepts[n], alpha=0.02, color='grey')

    # Area
    plt.fill_between(x_interp, y_interp_min, y_interp_max, alpha=area_alpha, color=col)

    # Outliers
    if show_outliers:
        for i in outlier_inds:
            plt.plot(x_mean[i], y_mean[i], '.', color='r', markersize=5,zorder=5)

    # Test
    if mask_good is None:
        corr = pearsonr(x_mean, y_mean)
    else:
        corr = pearsonr(x_mean[mask_good], y_mean[mask_good])

    # Faccio anche una regressione su tutto
    res = linregress(x_mean, y_mean)


    y_pred = np.mean(slopes) * x_mean + np.mean(intercepts)
    y_mean_mean = np.mean(y_mean)

    ss_res = np.sum((y_mean - y_pred)**2)
    ss_tot = np.sum((y_mean - y_mean_mean)**2)
    print(corr.statistic**2)
    # plt.title(f'Pearson r = {corr.statistic:.2f}, p = {corr.pvalue:.4f}')

    return corr.statistic, corr.pvalue
    # plt.title(f'R = {np.mean(corrcoeff):.2f}')



# def plot_decision_state(room_values, current_room, time, center):

#     ax = plt.gca()

#     room_names = ['N','E','S','W']
#     L = 1

#     for i in range(4):
#         if i == current_room:
#             fc = np.array([0.8,1,0.8])
#         else:
#             fc = 'w'

#         x_c = center[0] - 1.5 * L + i * L
#         y_c = center[1]
#         x = x_c - L/2
#         y = y_c - L/2
#         patch = patches.Rectangle((x, y), L, L, ec='k', fc=fc)
#         ax.add_patch(patch)

#         plt.text(x_c, y_c + L/2 + L/3, room_names[i], ha='center', va='center')
#         plt.text(x_c, y_c, room_values[i], ha='center', va='center')
#     plt.text(x_c + L/2 + L/3, y_c, f'T = {time}', ha='left', va='center')


def plot_decision_state(room_values, current_room, time, center):

    ax = plt.gca()

    room_names = ['N', 'E', 'S', 'W']
    L = 1

    for i in range(4):
        if i == current_room:
            fc = np.array([0.8, 1, 0.8])
        else:
            fc = 'w'

        x_c = center[0] - L/2 + L * (i % 2)
        y_c = center[1] - L/2 + L * (i // 2)
        x = x_c - L/2
        y = y_c - L/2
        patch = patches.Rectangle((x, y), L, L, ec='k', fc=fc)
        ax.add_patch(patch)

        # plt.text(x_c, y_c + L/2 + L/3, room_names[i], ha='center', va='center')
        plt.text(x_c, y_c, room_values[i], ha='center', va='center')
    plt.text(center[0], center[1] - L - L/4, f'${time}$', ha='center', va='top', fontsize=12)


def plot_decision_arrow(c1, c2, col='k'):

    L = 1.0
    c1 = c1 - np.array([0, L * 2])
    c2 = c2 + np.array([0, L])

    plt.arrow(c1[0],c1[1], c2[0]-c1[0], c2[1]-c1[1], width=0.05, color=col, length_includes_head=True)
    

def plot_decision_circle(c):

    ax = plt.gca()
    patch = patches.Circle((c[0],c[1]), 0.5, color='w', ec='k')
    ax.add_patch(patch)


def plot_decision_impossible(center):
     
    L = 2
    ax = plt.gca()

    x = center[0] - L/2 * 1.2
    y = center[1] - L/2 * 1
    patch = patches.Rectangle((x, y), L*1.2, L*1, ec='k', fc='w')
    ax.add_patch(patch)

    # plt.text(x_c, y_c + L/2 + L/3, room_names[i], ha='center', va='center')
    plt.text(center[0], center[1], 'Forbidden\nAction', ha='center', va='center', fontsize=11)

def plot_circle_arrow(c1, c2, col='k', from_circle=False, width=0.05):

    L = 1.0
    if from_circle:
        c1 = c1 - np.array([0, L/2])
        c2 = c2 + np.array([0, L])

        # Qui ci va il testo
        dx = np.abs(c2[0] - c1[0])
        c_avg = 0.3 * c1 + 0.7 * c2
        plt.text(c_avg[0] + 0.1 + 0.1 * dx, c_avg[1], '$50\%$', ha='left', va='center', fontsize=11)

    else:
        c1 = c1 - np.array([0, L * 2])
        c2 = c2 + np.array([0, L/2])

    plt.arrow(c1[0],c1[1], c2[0]-c1[0], c2[1]-c1[1], width=width, color=col, length_includes_head=True, head_width=0.15)
    
