# Silly Stats on Tips
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.multicomp as posthoc
import matplotlib.pyplot as graph

sns.set()


# Functions
def line(m, x, b):
    return m * x + b  # y = mx + b


# Get data
column_names = ['#', 'Tips', 'Hours', 'Date', '?']
data = pd.read_csv('01-06-2016-tipsee-12-13.csv', header=-1)
data.columns = column_names

# Add Tips per hour column
data['Tips per Hour'] = data['Tips'] / data['Hours']

dates = pd.to_datetime(data['Date'])
data['Day'] = pd.to_datetime(data['Date']).dt.dayofweek

'''
Time to run some tests!
'''

# Run linear regression
m, b, r_value, p_value, _ = stats.linregress(data['Hours'], data['Tips'])
x_line = [data['Hours'].min(), data['Hours'].max()]
y_line = [line(m, data['Hours'].min(), b), line(m, data['Hours'].max(), b)]

# Visualise Data
sns.jointplot(x='Hours', y='Tips', data=pd.DataFrame(data, columns=['Hours', 'Tips']))
graph.show()

# Which day has the best tips per hour rate?
day_array = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

bar_graph_data = pd.DataFrame(data, columns=['Day', 'Tips per Hour'])
bar_graph_data['Day'] = bar_graph_data['Day'].apply(lambda x: day_array[x])

sns.barplot(x='Day', y='Tips per Hour', data=bar_graph_data, palette='Blues_d')
graph.title('Which day has the best Tips per Hour Rate?')
graph.ylabel('Tips / Hour')
graph.show()

# Run ANOVA with Post Hoc test to figure which day is significantly different
post_test = posthoc.MultiComparison(data=data['Tips per Hour'], groups=data['Day'])
posthoc_test = post_test.tukeyhsd()
print(posthoc_test)

# Results all the means are equal... That's boring as fuck.
