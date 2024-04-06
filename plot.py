import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('stepsToGoal.csv', header=None, names=['Steps'])

#starting from the second row
data = data.iloc[1:]

plt.figure(figsize=(10, 6))

plt.plot(data.index, data['Steps'], marker='o', linestyle='-', color='b')
plt.title('Learning Progression ') 
plt.xlabel('Attempt Number')
plt.ylabel('Steps to Goal')
plt.grid(True)
plt.show()
