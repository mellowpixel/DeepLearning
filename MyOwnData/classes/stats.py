import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import time

# class Stats(Callback):
class Stats(Callback):

    def __init__(self):
        self.training = []
        self.test = {}
        self.epoch_start_time = 0


    def on_train_begin(self, logs=None):
        print("Training started")
        self.training = []
        

    def on_epoch_begin(self, epoch, logs=None):
        self.training.append({'n_samples': [], 'accuracy':[], 'loss':[], 'time':0})
        self.epoch_start_time = time.time()
    
    
    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            self.training[-1]['n_samples'].append(logs['batch']*logs['size'])
            self.training[-1]['accuracy'].append(logs['accuracy'])
            self.training[-1]['loss'].append(logs['loss'])
        else:
            print("* Logs not available. Batch:\n{}".format(batch))


    def on_epoch_end(self, epoch, logs=None):
        self.training[-1]['time'] = time.time() - self.epoch_start_time

    def on_train_end(self, logs=None):
        """  """


    def on_test_begin(self, logs=None):
        self.test = {'n_samples': [], 'accuracy':[]}

        # if logs is not None:
        #     keys = list(logs.keys())
        #     print("Start testing; got log keys: {}".format(keys))
        # else:
        #     print("* Logs not available (on_test_begin).")


    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            self.test['n_samples'].append(logs['batch']*logs['size'])
            self.test['accuracy'].append(logs['accuracy'])
        else:
            print("No test batch logs")


    def on_test_end(self, logs=None):
        """  """

    
    # *********  Training Accuracy Plot ********* #

    def show_training_stats(self, models_stats=None):
        plt.autumn()
        fig = plt.figure(figsize=(7,7), dpi=100)
        plt.yticks(np.arange(0, 1.2, 0.1))
        # fig.suptitle("Training accuracy")

        if models_stats is not None:
            # Stats of each model
            spi = 0 # subplot index
            n_subplots = len(models_stats.items()) # total number of subplots
            row_labels = []
            row_data = []
            for name, data in models_stats.items():
                spi += 1
                ax = fig.add_subplot(n_subplots, 1, spi)
                # Each epoch stats of each model
                table_col_labels = []
                cell_data = []
                for i, ep in enumerate(data['training']):
                    ax.plot('n_samples', 'accuracy', data=ep, label="Epoch " + str(i+1), linestyle='solid')
                    cell_data.append("{}%".format(round(ep['accuracy'][-1] * 100, 2)))
                    table_col_labels.append("Epoch " + str(i+1))

                row_labels.append(name + " training accuracy")
                row_data.append(cell_data)
                ax.set_ylabel("Accuracy")
                ax.legend()
                ax.grid(alpha=0.3, linewidth=1, ls='dashed')
                ax.set_title(str(name) + " Training Accuracy")
                ax.set_yticks(np.arange(0.0, 1.1, 0.1))
                ax.set_yticklabels(np.round(np.arange(0.0, 1.0, 0.1), 1))

                if spi == n_subplots:
                    spacer = -0.7
                else:
                    spacer = -0.5
                ax.table(cellText=[cell_data], rowLabels=["Accuracy"], colLabels=table_col_labels, loc='bottom', cellLoc='center', rowLoc='center', bbox=(0.1, spacer, 0.9, 0.3))

        if len(models_stats.items()) == 1:
            bottom = 0.4
        else:
            bottom = 0.29

        plt.subplots_adjust(hspace=0.98, bottom=bottom, top=0.9)
        plt.xlabel("Number of samples")

    

    # *********  Test Accuracy Plot ********* #

    def show_test_stats(self, models_stats=None):
        plt.figure()
        table_col_labels = []
        cell_data = []
        for name, data in models_stats.items():
            plt.plot('n_samples', 'accuracy', data=models_stats[name]['test'], label=name, linestyle='dotted')
            plt.ylabel("Accuracy")
            plt.title("Test Accuracy")
            plt.legend()
            plt.grid(alpha=0.3, linewidth=1, ls='dashed')
            plt.yticks(np.arange(0.0, 1.1, 0.1))
            table_col_labels.append(name)
            cell_data.append("{}%".format(round(data['test']['accuracy'][-1] * 100, 2)))
        
        plt.table(cellText=[cell_data], 
                  rowLabels=["Accuracy"], 
                  colLabels=table_col_labels, 
                  loc='bottom', 
                  cellLoc='center', 
                  rowLoc='center', 
                  bbox=(0.1, -0.5, 0.95, 0.3))

        plt.subplots_adjust(bottom=0.4, top=0.9)


    
    # *********  Training Time Bar Chart ********* #

    def training_performance(self, models_stats=None):
        models_times = []

        for name, data in models_stats.items():
            models_times.append({'name': name, 'time': [ m['time'] for m in data['training']]})

        fig = plt.figure()
        x_labels = ["epoch {}".format(i+1) for i in range(len(models_times[0]["time"]))] + ["Total"]
        bar_width = 0.36
        max_time = 0

        print("Models Times: ", models_times, "\nX Labels: ", x_labels)

        for m in models_times:
            ax = fig.add_subplot()
            tot_time = np.sum(m['time'])
            y_bars = m['time'] + [tot_time]
            if tot_time > max_time:
                max_time = tot_time
            bar_width *= -1
            bars = ax.bar(np.arange(len(x_labels)) + bar_width/2, y_bars, bar_width, label=m['name'])

            for b in bars:
                height = b.get_height()
                ax.annotate(str(round(height, 1)),
                            xy=(b.get_x() + b.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_title("Training Time")
        ax.set_ylabel("Seconds")
        ax.set_xticks([i for i in range(len(x_labels))])
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(alpha=0.3, linewidth=1, ls='dashed')
        fig.tight_layout()



    def words_count_distribution(self, df, column, title, show=True, percents=False, group_range=1):
        min = len(df[column][0])
        max = 0
        mean = 0
        sum = 0
        total_reviews = len(df[column])
        counter = {}

        for d in df[column]:
            l = len(d)
            
            if group_range > 1:
                mul_ten = np.ceil(l / group_range) * group_range
            else:
                mul_ten = l

            if mul_ten in counter:
                counter[mul_ten] += 1
            else:
                counter[mul_ten] = 1

            sum += l
            if l < min:
                min = l

            if l > max:
                max = l

            mean = sum / len(df[column])

        if percents == True:
            counter_sorted = { dist:np.round(total / total_reviews * 100) for dist, total in sorted(counter.items(), key=lambda item: item[1], reverse=True) }
        else:
            counter_sorted = { dist:total for dist, total in sorted(counter.items(), key=lambda item: item[1], reverse=True) }

        plt.figure()
        plt.title(title)
        plt.plot(list(counter_sorted.keys()), list(counter_sorted.values()), 'go', markersize=1, label="Review")
        plt.axvline(mean, c='m', label="Mean word count", linewidth=1)
        plt.xlabel("Words counts.")
        plt.ylabel("Number of reviews.")
        plt.legend()
        plt.grid(alpha=0.3, linewidth=1, ls='dashed')
        plt.xticks(np.arange(0, 2500, 100), rotation=45)
        # plt.yticks(np.arange(0, 10, 1))
        # plt.ylim([0, 10])
        print("\n-----------------------------------------------------")
        # print("* Words Count Distribution",counter_sorted)
        print("* Total Reviews: {}\n* Shortest Review: {}\n* Longesr_review: {}\n* Average reviwes: {}".format(len(df), min, max, mean))
        if show == True:
            plt.show()