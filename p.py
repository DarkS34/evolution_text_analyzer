import pandas as pd

# Leer el CSV original
df = pd.read_csv('icd_dataset.csv')

# Reordenar las columnas
df = df[['description', 'code']]

# Guardar el nuevo CSV
df.to_csv('invertido.csv', index=False)