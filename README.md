# Tool to Predict Used Car Prices

## Motivation
The inspiration for this project stems from the need to design a model capable of predicting the prices of used cars. This model can assist customers in determining if they are purchasing a car at a fair price.

## Description
This model utilizes a dataset containing the following features: Price, Year, Mileage, Model, Brand, City, and State. The information is sourced from the United States. The model is trained on machine learning algorithms that iteratively learn with each new piece of data added. A web page with a map interface allows users to input model, year, state, and mileage to receive a price estimate.

## Construction of the Model

### Tools
- Google Colab
- Kaggle
- Visual Studio
- GitHub

### Database Tools
- SQL

### Programming Languages
- HTML
- Java
- Python

### Libraries
- pandas
- numpy
- joblib
- sklearn.model_selection (train_test_split)
- sklearn.preprocessing (StandardScaler)
- sklearn.ensemble (RandomForestRegressor, GradientBoostingRegressor, VotingRegressor)
- xgboost (XGBRegressor)
- sklearn.metrics (mean_absolute_error, mean_squared_error, r2_score)


## Steps

1. **Import Libraries and Data**
2. **Data Loading and Cleaning**
3. **Data Splitting**
4. **Scaling Data**
5. **Define and Train the Voting Regressor with Weighted Models**
6. **Price Prediction Example**
7. **Test Prediction with 3 Cars Over 5 Years**


## Usage Instructions
Download all the files and place them in a folder together with the same name. Adjust the directions to pull the model, Java, data, and HTML with a local host so you can use the web page installed. Run in Python and use a browser to navigate the tool that provides car price predictions.

## Difficulties
Some difficulties encountered during the creation of this project included finding a suitable dataset. Initially, we used a dataset from India that worked fine with logistic regression, but the random forest did not work well with linear regression. Due to the lack of knowledge of the Indian market, this dataset was deemed unsuitable.

The second dataset we found had fewer columns but was very large (820K rows * 9 columns), making it difficult to work with in Google Colab and Visual Studio. For the scope of this project, we found a third dataset with 2.6k rows and more than 100 columns, resulting in better R² and model accuracy.

![image](https://github.com/user-attachments/assets/f5be3422-cdd2-453e-b21a-ebd7e7d30e90)


This is why this dataset was chosen.

## Database Information
The dataset was obtained from the following page: [Kaggle Dataset](https://www.kaggle.com/datasets/tymekurban/new-cars-usa-202223-dataset). The dataset is free to use, run, and modify.

## Tableau 
Complete Tableau project can be found on the following URL. This task was performed by the team mentioned in the Authors section:
https://public.tableau.com/views/FinalProject_17229013835720/Story1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Authors
- **Daniel Bardán** (@Wynton0535)
- **Rodrigo Figueroa** (@RoFigueroa18)
- **Pablo Morales** (@PabloUMD)
- **Carlos Valencia** (@carval9)
- **Jose Franco** (@jrobertofg)
