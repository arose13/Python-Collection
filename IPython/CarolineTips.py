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
graph.title('Tips Regression Rsq:{} p:{}'.format(round(r_value**2, 5), p_value))
graph.scatter(data['Hours'], data['Tips'])
graph.plot(x_line, y_line)
graph.xlabel('Hours')
graph.ylabel('Tips ($)')
graph.show()

# Which day is best?
day_range = [0, 1, 2, 3, 4, 5, 6]
day_array = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
means = []
stds = []

for day in range(len(day_range)):
    rows = data.where(data['Day'] == day)['Tips per Hour']

    mean = rows.mean()
    std = rows.std()

    means.append(mean)
    stds.append(std)

graph.title('Average Tips per Day')
graph.bar(day_range, means)
graph.scatter(data['Day'], data['Tips per Hour'], marker='x')
graph.ylabel('Mean Tips $')
graph.xlabel('Days')
graph.xticks(day_range, day_array)
graph.show()

# Run ANOVA with Post Hoc test to figure which day is significantly different
post_test = posthoc.MultiComparison(data=data['Tips per Hour'], groups=data['Day'])
posthoc_test = post_test.tukeyhsd()
print(posthoc_test)

# Results all the means are equal... That's boring as fuck.
