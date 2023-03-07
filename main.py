import pandas as pd

melbourne_data = pd.read_csv('/Users/abdullahsiddiqi/Documents/repos/IntroML-Kaggle/melb_data.csv')

melbourne_data.describe()

#Process:
## target (y)
## features (x)
## define the model
## Print the model
## or make model validation

#set the target for the ML model
y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melbourne_data[melbourne_features]

# define the model

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(x, y)


## the pridictions for the first 5 houses, these are the first 5 houses:
print(x.head())
# these are the predictions for the prices
print(melbourne_model.predict(x.head()))

## model validation
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(x)
mean_absolute_error(y, predicted_home_prices) 

from sklearn.model_selection import train_test_split

#split the data with random state so that it is the same every time
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

#define the model type again
melbourne_model = DecisionTreeRegressor()

#fit the model
melbourne_model.fit(train_x, train_y)

val_predictions = melbourne_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions)) # this is the is the out of sample error - is is around $250,000 in error

## now we have to try models that are better, more accurate
## Overfitting and Underfitting

## write a function to get mae
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    # in this case you'll see that 500 is the optimal number of leaves
# When you find the best decision tree you can use model = decisiontreeregressor(max_leaf_nodes = --, random_state= --)

## Random Forrests
# Similar to Decision trees but uses many different trees instead and then averaging 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)
melb_preds = forest_model.predict(val_x)
print(mean_absolute_error(val_y, melb_preds)) #Note that this is lower than before
