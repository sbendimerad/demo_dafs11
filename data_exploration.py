import pandas as pd
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target labels (species)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())

print(df.describe())

## First time git setup
# 1. Create a git repository on github
# 2. Create the repository on local machine 
# 3. echo "# dafs11 demo" >> README.md
# 4. git init (initialize the git repository on local machine in the reight directory)
# 5. git add README.md
# 6. git commit -m "first commit"
# 7. git branch -M main
# 8. git remote add origin "url of the repository"
# 9. git push -u origin main

## Pushing the changes to the repository
# git status
# git add data_exploration.py
# git commit -m "Add data exploration script"
# git push origin main