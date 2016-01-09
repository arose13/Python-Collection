import matplotlib.pyplot as graph
import seaborn as sns


sns.set()
data = sns.load_dataset('iris')
sns.pairplot(data=data, hue='species')
print(data)
graph.show()
