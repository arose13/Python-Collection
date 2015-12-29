# Adding Columns to a DataFrame
# Also check out this StackOverFlow answer...
# http://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-of-certain-column-is-nan
import numpy as np
import pandas as pd

# Prep Values
a = [1, 2, 3, 0]
b = [4, 5, 6, -3]
c = [7, 8, 9, -6]
d = [10, np.nan, 12, -12]
e = [13, 14, 15, -15]

# Make DataFrame with the data
df = pd.DataFrame()
df['alpha'] = a
df['bravo'] = b
df['charlie'] = c
df['delta'] = d
df['echo'] = e

print('\nOriginal DataFrame')
print(df)

# Make each row an array
f = df.values
print('\nNumPy array')
print(f)

# Remove row with NaN value
g = df.dropna().values
g = np.array(g)
print('\nNumPy array with the NaN rows removed')
print(g)

# Let's just sort g but the numbers in the first row just to see what it says
g_sort = g[g[:, 0].argsort()]
print('\nSorted by first column')
print(g_sort)

h = g.T  # Transpose. We're going to select the first 60% of rows
print('\nNumPy array Transposed')
print(h)

# Select the first 60% of rows
h60 = h[0:(round(0.6 * len(h)))]
print('\nFirst 60% of the transposed rows')
print(h60)

h40 = h[(round(0.6 * len(h))): len(h)]
print('\nLast 40% of the transposed rows')
print(h40)

print('\nShifting values in a single column in a particular direction')
i1 = [1, 2, 3]
i2 = [1, 2, 3]
i3 = [1, 2, 3]

i = np.column_stack((i1, i2, i3))
print('Before\n', i)

i[:, 0] = np.roll(i[:, 0], -1)
print('After\n', i)
