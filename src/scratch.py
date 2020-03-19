import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTK3Agg')

with open('acc.csv', newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x, acc, std = [], [], []
    for row in csv_reader:
        if line_count == 0:
            print(f'Row is {", ".join(row)}')
        else:
            x.append(int(row[0]))
            acc.append(float(row[1]))
            std.append(float(row[2]))
        line_count += 1

plt.plot(x, acc)
plt.title('Accuracy over epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()

