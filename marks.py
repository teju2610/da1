import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df = pd.read_csv("marks.csv")
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['percentage'] = (df['marks_obtained'] / df['total_marks']) * 100
plt.figure(figsize=(10, 5))
plt.bar(df['student_name'], df['percentage'])
plt.xlabel('Student')
plt.ylabel('Percentage')
plt.title('Percentage vs Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




minmax = MinMaxScaler()
minmax_scaled = minmax.fit_transform(df)

standard = StandardScaler()
standard_scaled = standard.fit_transform(df)

minmax_df = pd.DataFrame(minmax_scaled, columns=df.columns)
standard_df = pd.DataFrame(standard_scaled, columns=df.columns)

print(minmax_df)
print(standard_df)