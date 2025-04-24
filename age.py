import pandas as pd
from datetime import datetime
 
df = pd.read_csv("marks.csv")
df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
df['age'] = datetime.now().year - df['dob'].dt.year

print(df)