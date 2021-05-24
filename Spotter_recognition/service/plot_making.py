import matplotlib.pyplot as plt
import numpy as np


def make_plot(prediction_data, thershold):
    predictions = [pred[0] for pred in prediction_data]
    fig = plt.figure(figsize=(8,8))
    plt.clf()
    plt.plot(predictions, label='Spotter prediction')
    x = np.arange(0, len(predictions), 1)
    y = [thershold] * len(predictions)
    plt.plot(x, y, color='red', linestyle='dashed', label='Thershold spotter') 
    prev_label = ""
    max_p = 0
    max_index = 0
    for index, x in enumerate(prediction_data):
        if x[0] > thershold:
            label = x[1]
            if label == prev_label:
                if x[0] > max_p:
                    max_p, max_index = x[0], index 
                continue
            elif prev_label != "" and prev_label != "unknown":
                plt.plot(max_index, max_p, 'ro', markersize=10, markerfacecolor='blue')
                plt.text(max_index, max_p, prev_label, fontsize=15)
            prev_label = label
            max_p = x[0]
            max_index = index
        else:
            if prev_label != "" and prev_label != 'unknown':
                plt.plot(max_index, max_p, 'ro', markersize=10, markerfacecolor='blue')
                plt.text(max_index, max_p, prev_label, fontsize=15)
            prev_label = ""
            max_p = 0
            max_index = 0

    plt.title('Model prediction')
    plt.ylabel('Softmax prediction value')
    plt.xlabel('Time')
    plt.legend(loc='lower right')
    plt.savefig("model_prediction.jpg", facecolor=fig.get_facecolor(), transparent=True)
