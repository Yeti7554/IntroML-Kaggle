import pandas as pd

melbourne_data = pd.read_csv('/Users/abdullahsiddiqi/Documents/repos/IntroML-Kaggle/melb_data.csv')

melbourne_data.describe()

#Process:
## target (y)
## features (x)
## define the model

#set the target for the ML model
y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melbourne_data[melbourne_features]

# define the model

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(x, y)