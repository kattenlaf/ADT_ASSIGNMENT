import pandas as pd

raw_data = {
    'X1' : [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166],
    'X2' : [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72],
    'X3' : [173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217]
}

df = pd.DataFrame(raw_data, columns=['X1', 'X2', 'X3'])
df.to_csv('Data_Set.csv')