import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='pubmed_NSGAT.json', type=str)
args = parser.parse_args()

with open(args.filename, 'r') as f:
    data = json.load(f)

test_acc = [item['test_acc'] for item in data]
times = [item['time'] for item in data]
cumulative_time = [sum(times[:i+1]) for i in range(len(times))]

plt.figure(figsize=(10, 5))
plt.plot(cumulative_time, test_acc, label='Test Accuracy', marker='o', linestyle='-', color='green')
plt.title('Test Accuracy Over Cumulative Time')
plt.xlabel('Cumulative Time (s)')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('test_accuracy_over_time.jpg')
plt.show()
