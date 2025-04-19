import pandas as pd

df = pd.read_csv("iris_with_errors.csv", na_values=['-', 'NA'])
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')

def a():
    missing_data = df.isnull().sum()
    print("\nLiczba brakujących danych w każdej kolumnie:")
    print(missing_data)
    print(f"\nŁączna liczba brakujących danych: {df.isnull().sum().sum()}")

    print("\nStatystyki bazy danych:")
    print(df.describe())

    print("\nInformacje o kolumnach:")
    print(df.info())

def b():
    numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    
    print("\nWartości spoza zakresu (0; 15) przed korektą:")
    for col in numeric_columns:
        invalid_values = df[(df[col] <= 0) | (df[col] >= 15)][col]
        if not invalid_values.empty:
            print(f"\n{col}:")
            print(invalid_values)
    
    for col in numeric_columns:
        valid_values = df[(df[col] > 0) & (df[col] < 15)][col]
        median_value = valid_values.median()
        
        df.loc[(df[col] <= 0) | (df[col] >= 15), col] = median_value
    
    print("\nPo korekcie - statystyki dla kolumn numerycznych:")
    print(df[numeric_columns].describe())
    
    print("\nSprawdzenie czy pozostały wartości spoza zakresu (0; 15):")
    for col in numeric_columns:
        invalid_count = df[(df[col] <= 0) | (df[col] >= 15)][col].count()
        print(f"{col}: {invalid_count} wartości spoza zakresu")

def c():
    correct_names = {'Setosa', 'Versicolor', 'Virginica'}
    
    print("\nUnikalne nazwy gatunków przed korektą:")
    print(df['variety'].unique())
    
    def correct_variety_name(name):
        name = str(name).title()
        
        corrections = {
            'Setosa': 'Setosa',
            'Versicolour': 'Versicolor',
            'Versicolor': 'Versicolor',
            'Virginica': 'Virginica'
        }
        
        return corrections.get(name, None)
    
    df['variety'] = df['variety'].apply(correct_variety_name)
    
    print("\nUnikalne nazwy gatunków po korekcie:")
    print(df['variety'].unique())
    
    incorrect_names = df[~df['variety'].isin(correct_names)]
    if not incorrect_names.empty:
        print("\nPozostałe niepoprawne nazwy gatunków:")
        print(incorrect_names['variety'].unique())

a()
b()
c()