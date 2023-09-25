# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# read the employees.csv file
chatterbox = pd.read_csv('employees.csv')

# select the rows with values '0000' for Year_of_Birth and replace that with 0
chatterbox['Year_of_Birth'] = chatterbox['Year_of_Birth'].replace("'0000'", np.nan)

# change the Date_Joined column, Date_Resigned, Inactive_Date to datetime format fill suitable values for invalid dates
chatterbox['Date_Joined'] = pd.to_datetime(chatterbox['Date_Joined'], errors='coerce')
chatterbox['Date_Resigned'] = pd.to_datetime(chatterbox['Date_Resigned'], errors='coerce')
chatterbox['Inactive_Date'] = pd.to_datetime(chatterbox['Inactive_Date'], errors='coerce')


# if status is active make Date_Resigned and Inactive_Date as NaT
chatterbox.loc[chatterbox['Status'] == 'Active', ['Date_Resigned', 'Inactive_Date']] = np.nan

# if Date_Resigned is NaT and Status is Inactive then fill Date_Resigned with Inactive_Date
chatterbox.loc[(chatterbox['Date_Resigned'].isna()) & (chatterbox['Status'] == 'Inactive'), 'Date_Resigned'] = chatterbox['Inactive_Date']

# change Titile of all males as Mr. and females as Ms.
chatterbox.loc[chatterbox['Gender'] == 'Male', 'Title'] = 'Mr.'
chatterbox.loc[chatterbox['Gender'] == 'Female', 'Title'] = 'Ms.'


df1 = chatterbox.filter(['Gender', 'Religion_ID', 'Employment_Type', 'Designation', 'Year_of_Birth'], axis=1)

# one hot encoding for df1
df1 = pd.get_dummies(df1, columns=['Gender', 'Employment_Type', 'Designation'])

# knn imputer for df1 handle missing values
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df1 = pd.DataFrame(imputer.fit_transform(df1),columns = df1.columns)

chatterbox['Year_of_Birth'] = df1['Year_of_Birth'].astype(int)

# Subset the desired columns
df2 = chatterbox.filter(['Gender', 'Religion_ID', 'Employment_Type', 'Designation', 'Year_of_Birth', 'Marital_Status'], axis=1)

# Perform label encoding on 'Marital_Status'
le = LabelEncoder()
df2['Marital_Status'] = le.fit_transform(df2['Marital_Status'])

# replace value 2 in Marital_Status with NaN
df2['Marital_Status'] = df2['Marital_Status'].replace(2, np.nan)

# one hot encoding for df2 columns gender, employment_type, designation, status
df2 = pd.get_dummies(df2, columns=['Gender', 'Employment_Type', 'Designation'])


imputer = KNNImputer(n_neighbors=1)
df2 = pd.DataFrame(imputer.fit_transform(df2),columns = df2.columns)

# replace marital status with Married if 0 else Single
df2['Marital_Status'] = df2['Marital_Status'].replace(0, 'Married')
df2['Marital_Status'] = df2['Marital_Status'].replace(1, 'Single')

chatterbox.Marital_Status = df2.Marital_Status

chatterbox.Date_Resigned.fillna('\\N', inplace=True)
chatterbox.Inactive_Date.fillna('\\N', inplace=True)

# save cleaned data to csv file
chatterbox.to_csv('employee_preprocess_200240M.csv', index=False)
