import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Membaca data dari file CSV
df = pd.read_csv('datasets_triase.csv')

# Replace 'L' with 0 and 'P' with 1 in the 'Jenis Kelamin' column
df['Jenis Kelamin'] = df['Jenis Kelamin'].replace({'L': 0, 'P': 1})

# Mengganti nilai 'P1' hingga 'P150' dengan nilai 0 pada kolom 'ID'
df['ID'] = df['ID'].replace(to_replace=r'^P\d+$', value=0, regex=True)

df['Triage'] = df['Triage'] = df['Triage'].replace({'Merah': 1, 'Kuning': 2, 'Hijau': 3})

df.drop(columns=['ID'], inplace=True)

# Memisahkan fitur dan label
X = df.drop('Triage', axis=1)
y = df['Triage']

# Melatih model
model = DecisionTreeClassifier()
model.fit(X, y)

# Menyimpan model ke dalam file model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
