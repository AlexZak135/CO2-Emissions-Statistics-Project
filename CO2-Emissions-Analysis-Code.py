# Title: CO2 Emissions Statistical Analysis
# Author: Alexander Zakrzeski
# Date: March 3, 2023

# Import the necessary libraries and modules

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data

df1 = pd.read_csv("CO2_Data.csv")

##### Part A - Data Preprocessing ##### 

# Consolidate the column names 

df1.columns = (df1
               .columns
               .str.lower()
               .str.replace(" ", "_"))

# Take the following steps: 
  # 1. Modify certain values in columns
  # 2. Create a new column
  # 3. Filter based on the set condition 
  # 4. Rename specific columns
  # 5. Drop a column
    
df1 = (df1
       .assign(make = df1["make"] 
               .apply(lambda x: x.title() if x not in 
                      ["BMW", "FIAT", "GMC", "MINI", "SRT"] 
                      else x),
               continent = df1["make"]
               .apply(lambda x: "North America" if x in 
                      ["Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", 
                       "Ford", "GMC", "Jeep", "Lincoln", "Ram", "SRT"] 
                      else "Europe" if x in 
                      ["Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", 
                       "Bugatti", "FIAT", "Jaguar", "Lamborghini", "Land Rover", 
                       "Maserati",  "Mercedes-Benz", "MINI", "Porsche", 
                       "Rolls-Royce", "Smart", "Volkswagen", "Volvo"] 
                      else "Asia"), 
               transmission = df1["transmission"]
               .apply(lambda x: "Automatic" if 
                      re.match(r"^A\d*$", x) 
                      else "Automated Manual" if 
                      re.match(r"^AM\d*$", x) 
                      else "Automatic With Select Shift" if 
                      re.match(r"^AS\d*$", x) 
                      else "Continuously Variable" if 
                      re.match(r"^AV\d*$", x) 
                      else "Manual"),
               fuel_type = df1["fuel_type"]
               .apply(lambda x: "Diesel" if x in "D"  
                      else "Ethanol" if x in "E"  
                      else "Natural Gas" if x in "N"  
                      else "Regular" if x in "X"  
                      else "Premium"), 
               co2_emissions_bin = df1["co2_emissions(g/km)"] 
               .apply(lambda x: 1 if x > df1["co2_emissions(g/km)"].median() 
                      else 0)) 
       .query("fuel_type != 'Natural Gas'") 
       .rename(columns = {"engine_size(l)": "engine_size",
                          "fuel_consumption_comb_(l/100_km)": "consumption",
                          "co2_emissions(g/km)": "co2_emissions"})
       .drop(columns = "vehicle_class"))

##### Part B - Exploratory Data Analysis #####

# Generate the necessary descriptive statistics

list1 = ["co2_emissions", "consumption", "engine_size"] 

ds1 = (df1
       [list1] 
       .agg(["count", "mean", "std", "min", "max"]))

ds1.loc["med"] = (df1.median())

ds1.loc["mode"] = df1.mode().iloc[0]

ds1 = (ds1 
       .T
       [["count", "mean", "std", "med", "mode", "min", "max"]])

# Set the style of all plots

sns.set_style("whitegrid")

# Create a histogram displaying the distribution
 
sns.histplot(data = df1, x = "co2_emissions", bins = 20, kde = True, 
             color = "#679BE4")
plt.title("Distribution of CO2 Emissions", fontsize = 13)
plt.xlabel("CO2 Emissions (g/km)", fontsize = 11)
plt.ylabel("Frequency", fontsize = 11)
sns.despine(left = True, right = True)
plt.gca().xaxis.grid(False)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()

# Create a histogram displaying the distribution

sns.histplot(data = df1, x = "engine_size", bins = 20, kde = True,
             color = "#679BE4")
plt.title("Distribution of Engine Size", fontsize = 13)
plt.xlabel("Engine Size (L)", fontsize = 11)
plt.ylabel("Frequency", fontsize = 11)
sns.despine(left = True, right = True)
plt.gca().xaxis.grid(False)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()

# Get the Pearson correlation coefficients

cor1 = (df1
        [list1] 
        .corr(method = "pearson"))

# Create a scatter plot displaying the relationship between the variables  

sns.lmplot(data = df1, x = "engine_size", y = "co2_emissions", 
           line_kws = {"color": "black"}, scatter_kws = {"color": "#9ED1C2"}) 
plt.title("Relationship Between CO2 Emissions and Engine Size", fontsize = 13)
plt.xlabel("Engine Size (L)", fontsize = 11)
plt.ylabel("CO2 Emissions (g/km)", fontsize = 11)
sns.despine(left = True, right = True)
plt.show()

# Create a new list

list2 = list1 + ["cylinders"]

# Get the Spearman rank correlation coefficients

cor2 = (df1
        [list2]
        .corr(method = "spearman"))

# Get the point-biserial correlation coefficients

cor3 = stats.pointbiserialr(df1["engine_size"], df1["co2_emissions_bin"])

cor4 = stats.pointbiserialr(df1["consumption"], df1["co2_emissions_bin"])

# Create a function to perform a chi-square test

def chi2_output(df, x, y):
    df = df.astype({x.name: str, y.name: str})
    ct = pd.crosstab(df[x.name], df[y.name])
    chi2, p_value, dof, expected = chi2_contingency(ct)
    print(f"{x.name} {y.name} - p-value: {p_value}")

# Loop through the different x values

for x in ["continent", "fuel_type", "transmission"]:
    chi2_output(df1, df1[x], df1["co2_emissions_bin"])
    print("\n") 

# Identify outliers using the IQR method

q1 = df1["co2_emissions"].quantile(0.25)
q3 = df1["co2_emissions"].quantile(0.75)
iqr1 = q3 - q1
lower_bound1 = q1 - (1.5 * iqr1)
upper_bound1 = q3 + (1.5 * iqr1)

outliers1 = (df1 
             [["make", "model", "co2_emissions"]] 
             [(df1["co2_emissions"] < lower_bound1) | 
              (df1["co2_emissions"] > upper_bound1)]
             .sort_values(by = "co2_emissions", ascending = False))

# Remove the outliers using the IQR method and create a new column

df1 = (df1 
       [(df1["co2_emissions"] >= lower_bound1) & 
        (df1["co2_emissions"] <= upper_bound1)]
       .assign(engine_size_squared = np.power(df1["engine_size"], 2)) 
       .drop(columns = ["make", "model"]))

# Create a function to perform a one-way ANOVA

def aov_output(df, x, y): 
    formula = f"{y.name} ~ {x.name}"
    aov = ols(formula, data = df).fit()
    print(sm.stats.anova_lm(aov, typ = 2))
    print(pairwise_tukeyhsd(y, x))

# Loop through the different x values

for x in ["continent", "fuel_type", "transmission"]:
    aov_output(df1, df1[x],  df1["co2_emissions"])
    print("\n")  
    
##### Part C - Statistical Modeling ##### 

# Convert categorical variables to dummy variables

df2 = (df1
       .join(pd.get_dummies(df1[["continent", "fuel_type", "transmission"]], 
                            prefix = "", prefix_sep = ""))
       .drop(columns = ["continent", "Asia", "fuel_type", "Premium", 
                        "transmission", "Manual"]))

# Consolidate the column names 

df2.columns = (df2
               .columns
               .str.lower()
               .str.replace(" ", "_"))

### Polynomial Regression

# Take the following steps:
  # 1. Fit the model
  # 2. Generate the model's residuals
  # 3. Produce residuals vs. fitted plot
  # 4. Produce a histogram of the residuals
  # 5. Use custom function to get VIFs
  # 6. Generate the output of the model

lm1 = smf.ols(formula = "co2_emissions ~ engine_size + engine_size_squared + \
                         europe + north_america + diesel + ethanol + \
                         regular + automatic + automatic_with_select_shift + \
                         automated_manual + continuously_variable", 
                         data = df2).fit()

resid1 = lm1.resid

sns.residplot(x = lm1.fittedvalues, y = resid1, 
              scatter_kws = {"color": "#9ED1C2"})           
plt.title("Residuals vs. Fitted Values Plot", fontsize = 13)
plt.xlabel("Fitted Values", fontsize = 11)
plt.ylabel("Residuals", fontsize = 11)
sns.despine(left = True, right = True)
plt.show()

sns.histplot(x = resid1, bins = 20, kde = True, 
             color = "#679BE4")
plt.title("Distribution of the Residuals", fontsize = 13)
plt.xlabel("Residuals", fontsize = 11)
plt.ylabel("Frequency", fontsize = 11)
sns.despine(left = True, right = True)
plt.gca().xaxis.grid(False)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()

def vif_output(data): 
    columns = sm.add_constant(data)
    vifs = [variance_inflation_factor(columns.values, i) 
            for i in range(columns.shape[1])]
    series = pd.Series(vifs[1:], index = data.columns)
    return series

vifs1 = vif_output(df2
                   .drop(columns = ["consumption", "cylinders", 
                                    "co2_emissions_bin", "co2_emissions",
                                    "engine_size_squared"]))

lm1.summary()

### Logistic Regression 

# Take the following steps:
  # 1. Add a constant
  # 2. Create a scatter plot
  # 3. Fit the model
  # 4. Use custom function to get VIFs
  # 5. Generate the output of the model

df2 = sm.add_constant(df2)
 
sns.regplot(data = df2, x = "engine_size", y = "co2_emissions_bin", 
            logistic = True, ci = None,
            line_kws = {"color": "black"}, scatter_kws = {"color": "#9ED1C2"})
plt.title("Relationship Between Binary CO2 Emissions and Engine Size", 
          fontsize = 13)
plt.xlabel("Engine Size (L)", fontsize = 11)
plt.ylabel("Binary CO2 Emissions", fontsize = 11)
sns.despine(left = True, right = True)
plt.show()

glm1 = sm.Logit.from_formula("co2_emissions_bin ~ const + consumption + \
                              engine_size + europe + north_america + diesel + \
                              ethanol + regular + automatic + \
                              automatic_with_select_shift + automated_manual + \
                              continuously_variable", 
                              data = df2).fit()
                         
vifs2 = vif_output(df2 
                   .drop(columns = ["const", "cylinders", 
                                    "europe", "north_america", 
                                    "co2_emissions_bin", "co2_emissions",
                                    "engine_size_squared"])) 

glm1.summary()

odds_ratios1 = np.exp(glm1.params)