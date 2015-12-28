# Adding Columns to a DataFrame
# Also check out this StackOverFlow answer...
# http://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-of-certain-column-is-nan
import pandas as pd

# Prep Values
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
d = [10, None, 12]
e = [13, 14, 15]

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
print('\nNumPy array with the NaN rows removed')
print(g)

h = g.T  # Transpose. We're going to select the first 60% of rows
print('\nNumPy array Transposed')
print(h)

# Select the first 60% of rows
h60 = h[0:(round(0.6 * len(h)))]
print('\nFirst 60% of the transposed rows')
print(h60)
