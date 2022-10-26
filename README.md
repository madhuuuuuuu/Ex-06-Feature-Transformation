# Ex-06-Feature-Transformation
AIM:

To read the given data and perform Feature Transformation process and save the data to the file.

EXPLAINATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM:

STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file

CODE AND OUTPUT:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("C:\Users\lenovo\Documents\Data_to_Transform.csv")

df

![image](https://user-images.githubusercontent.com/95408668/198084192-b3394a18-aa2e-47b3-b3d5-98b3a706ce01.png)


df.head()

![image](https://user-images.githubusercontent.com/95408668/198084098-b16d1ac7-2663-4f16-9aac-20e3f5592aab.png)

df.isnull().sum()

![image](https://user-images.githubusercontent.com/95408668/198084023-51066bd1-a5aa-4349-b50c-e69995ffd6e1.png)

df.info()

![image](https://user-images.githubusercontent.com/95408668/198083953-8040c125-fe15-4702-8612-568dbfd743aa.png)

df.describe()

![image](https://user-images.githubusercontent.com/95408668/198083865-79dcc6cd-1438-4dc1-be9d-12f284426824.png)

df1 = df.copy()

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083760-512c1105-d053-475c-a7e5-705eed19e4ef.png)

sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083683-cb76e399-de17-4853-a60d-f826650790a8.png)

sm.qqplot(df1.ModeratePositivSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083578-ec015fc1-36f7-4f81-9cac-9e9d1090dc51.png)

sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083513-f913d6d7-2683-4385-ad3d-30870e611eaf.png)

df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083428-5aa8ab92-668e-485d-91ac-5c9bd6d59341.png)

df2 = df.copy()

df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083302-61a5e289-06b2-484c-af61-ee74877d6d2c.png)

df3 = df.copy()

df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083201-a2b6a42b-1362-4b71-83df-a46fe5c08571.png)

df4 = df.copy()

df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositivSkew)

sm.qqplot(df4.ModeratePositivSkew,fit=True,line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083114-6a61ddf2-e23c-408b-9d8d-623bf75b20c1.png)

from sklearn.preprocessing import PowerTransformer

trans = PowerTransformer("yeo-johnson")

df5 = df.copy()

df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198083010-746696d2-b484-4b72-bab8-d0b34dad4bc5.png)

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution = 'normal')

df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')

plt.show()

![image](https://user-images.githubusercontent.com/95408668/198082900-03d73a18-cae1-4360-9883-b305c7d29701.png)

RESULT:

Thus the Feature Transformation for the given datasets had been executed successfully.
