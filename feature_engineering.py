def add_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df