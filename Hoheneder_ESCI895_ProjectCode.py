#ESCI895 Lab 10 Python Script
#Script By: Tim Hoheneder, University of New Hampshire
#Date Created: 1 December 2021

#%% Description of Dataset and Purpose of Code: 

#Description: 
# I'll Fill This Portion in Later

#%% Run "Example Code for My Purposes:

#Install Libraries and !Pip: 
import pandas as pd
import datetime
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt

#%% Importing Raw Datafiles: 

#Importing Data Files: 
albright_file= 'Albright.txt'
blackwater_file= 'Blackwater.txt'
bowden_file= 'Bowden.txt'
hendricks_file = 'Hendricks.txt'
parsons_file= 'Parsons.txt'
rockville_file= 'Rockville.txt'

#%% Importing DataFrames: 

#Albright:
df_albright= pd.read_table(albright_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_albright= df_albright.drop(columns={"5s", "15s", "6s", "10s", "10s.1"})
#Rename Columns: 
df_albright= df_albright.rename(columns={"14n": "Discharge (cfs)"})
df_albright= df_albright.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_albright.interpolate(method = 'linear', inplace = True)

#Blackwater: 
df_blackwater= pd.read_table(blackwater_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_blackwater= df_blackwater.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_blackwater= df_blackwater.rename(columns={"14n": "Discharge (cfs)"})
df_blackwater= df_blackwater.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_blackwater.interpolate(method = 'linear', inplace = True)
    
#Bowden: 
df_bowden= pd.read_table(bowden_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_bowden= df_bowden.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_bowden= df_bowden.rename(columns={"14n": "Discharge (cfs)"})
df_bowden= df_bowden.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_bowden.interpolate(method = 'linear', inplace = True)
    
#Hendricks: 
df_hendricks= pd.read_table(hendricks_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_hendricks= df_hendricks.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_hendricks= df_hendricks.rename(columns={"14n": "Discharge (cfs)"})
df_hendricks= df_hendricks.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_hendricks.interpolate(method = 'linear', inplace = True)  
    
#Parsons: 
df_parsons= pd.read_table(parsons_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_parsons= df_parsons.drop(columns={"5s", "15s", "6s", "10s", "10s.1"})
#Rename Columns: 
df_parsons= df_parsons.rename(columns={"14n": "Discharge (cfs)"})
df_parsons= df_parsons.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_parsons.interpolate(method = 'linear', inplace = True)
    
#Rockville: 
df_rockville= pd.read_table(rockville_file, delimiter="\t", comment='#', 
                   header=1, parse_dates=['20d'], index_col=['20d'], 
                   na_values= [9999, -0.0, -0.1])
#Drop Columns: 
df_rockville= df_rockville.drop(columns={"5s", "15s", "6s", "10s"})
#Rename Columns: 
df_rockville= df_rockville.rename(columns={"14n": "Discharge (cfs)"})
df_rockville= df_rockville.rename(columns={"14n.1": "Stage (ft)"})
#Fill NaN Data: 
df_rockville.interpolate(method = 'linear', inplace = True)
    
#%% Creation of Z-Score Columns: 

#Create DataFrame List for Looping: 
dflist= [df_albright, df_blackwater, df_bowden, df_hendricks, df_parsons, df_rockville]

#Albright: 
df_albright['Z-Score Q']= (df_albright['Discharge (cfs)'] - 
                           df_albright['Discharge (cfs)'].mean()
                           ) / df_albright['Discharge (cfs)'].std()

#Blackwater: 
df_blackwater['Z-Score Q']= (df_blackwater['Discharge (cfs)'] - 
                           df_blackwater['Discharge (cfs)'].mean()
                           ) / df_blackwater['Discharge (cfs)'].std()

#Bowden: 
df_bowden['Z-Score Q']= (df_bowden['Discharge (cfs)'] - 
                           df_bowden['Discharge (cfs)'].mean()
                           ) / df_bowden['Discharge (cfs)'].std()

#Hendricks: 
df_hendricks['Z-Score Q']= (df_hendricks['Discharge (cfs)'] - 
                           df_hendricks['Discharge (cfs)'].mean()
                           ) / df_hendricks['Discharge (cfs)'].std()
    
#Parsons: 
df_parsons['Z-Score Q']= (df_parsons['Discharge (cfs)'] - 
                           df_parsons['Discharge (cfs)'].mean()
                           ) / df_parsons['Discharge (cfs)'].std()

#Rockville: 
df_rockville['Z-Score Q']= (df_rockville['Discharge (cfs)'] - 
                           df_rockville['Discharge (cfs)'].mean()
                           ) / df_rockville['Discharge (cfs)'].std()    

#%% Calculate Total Discharge by Watershed Area Drainage: 

#Define Watershed Area: 
watershed_area= 1422 #sq-mi

#Calculate Discharge Equivalence in cm/hr: 
#Albright:
df_albright['Discharge (cm/hr)']= (df_albright['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)
    
#Blackwater: 
df_blackwater['Discharge (cm/hr)']= (df_blackwater['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)

#Bowden: 
df_bowden['Discharge (cm/hr)']= (df_bowden['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)    

#Hendricks: 
df_hendricks['Discharge (cm/hr)']= (df_hendricks['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)
    
#Parsons:
df_parsons['Discharge (cm/hr)']= (df_parsons['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)
    
#Rockville: 
df_rockville['Discharge (cm/hr)']= (df_rockville['Discharge (cfs)']/watershed_area * (1/5280**2) * 30.48 * 3600)

#%% Plotting Initial Time Series Curves: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright['Discharge (cm/hr)'], ',', linestyle='-', color='navy', 
         label='Albright')
#Blackwater: 
ax1.plot(df_blackwater['Discharge (cm/hr)'], ',', linestyle='-', color='grey', 
         label='Blackwater')
#Bowden: 
ax1.plot(df_bowden['Discharge (cm/hr)'], ',', linestyle='-', color='dodgerblue', 
         label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks['Discharge (cm/hr)'], ',', linestyle='-', color='maroon', 
         label='Hendricks')
#Parsons: 
ax1.plot(df_parsons['Discharge (cm/hr)'], ',', linestyle='-', color='orange', 
         label='Parsons')
#Rockville: 
ax1.plot(df_rockville['Discharge (cm/hr)'], ',', linestyle='-', color='darkgreen', 
         label='Rockville')

#Axis Formatting: 
ax1.set_ylim(bottom = 0)
ax1.set_xlim(df_albright.index[0], df_albright.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Discharge Curves for Cheat River Watershed', fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75)) 
    
#%% Plotting Initial Time Series Curve Z-Scores: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright['Z-Score Q'], ',', linestyle='-', color='navy', label='Albright')
#Blackwater: 
ax1.plot(df_blackwater['Z-Score Q'], ',', linestyle='-', color='grey', label='Blackwater')
#Bowden: 
ax1.plot(df_bowden['Z-Score Q'], ',', linestyle='-', color='dodgerblue', label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks['Z-Score Q'], ',', linestyle='-', color='maroon', label='Hendricks')
#Parsons: 
ax1.plot(df_parsons['Z-Score Q'], ',', linestyle='-', color='orange', label='Parsons')
#Rockville: 
ax1.plot(df_rockville['Z-Score Q'], ',', linestyle='-', color='darkgreen', label='Rockville')

#Axis Formatting: 
ax1.set_xlim(df_albright.index[0], df_albright.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Z-Scored Discharge Curves for Cheat River Watershed', 
             fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75)) 

#%% Hydrograph Seperation Function: 

#Define Function:
def hydrograph_sep(totalq, watershed):

    #Find totalq: 
    totalq['Diff'] = totalq['Discharge (cm/hr)'].diff()
    
    #Find antecedent discharge and date using 0.0001 threshold for difference between events
    global antQ_date
    antQ = (totalq.loc[totalq['Diff'] > 0.000104, 'Discharge (cm/hr)'])
    antQ_date = antQ.index[0]
    antQ_val = round(antQ[0], 3)
    
    #Find Peak Discharge Date: 
    peakQ_date = totalq['Discharge (cm/hr)'].idxmax()
    peakQ = totalq['Discharge (cm/hr)'].max()   
    
    #Calculate Event Duration:
    N = 0.82*(watershed*1e-6)**0.2
    #Calculate End of Event: 
    global end_of_event
    end_of_event = peakQ_date + datetime.timedelta(days = N)
    
    #Calculate Ending Discharge Value: 
    end_Q = totalq.iloc[totalq.index.get_loc(end_of_event,method='nearest'), 1]
    
    #Create baseQ Dataframe:
    baseQ = totalq[['Discharge (cm/hr)']].copy()
    
    #Calculate Base Discharge Curve Before Peak: 
    slope1, intercept1= np.polyfit(totalq.loc[totalq.index < antQ_date].index.astype('int64')
                                /1E9, totalq.loc[totalq.index < antQ_date, 
                                                'Discharge (cm/hr)'], 1) 
    #Append Data Before Peak: 
    baseQ.loc[antQ_date:peakQ_date,"Discharge (cm/hr)"] = slope1 * (totalq.loc[antQ_date:peakQ_date].index.view('int64')/1e9) + intercept1
    
    #Calculate Base Discharge Curve After Peak: 
    slope2, intercept2= np.polyfit([peakQ_date.timestamp(), end_of_event.timestamp()], 
                               [baseQ.loc[peakQ_date, 'Discharge (cm/hr)'], end_Q], 1)
    #Append Data After Peak: 
    baseQ.loc[peakQ_date:end_of_event,"Discharge (cm/hr)"] = slope2 * (totalq.loc[peakQ_date:end_of_event].index.view('int64')/1e9) + intercept2
    
    #Append BaseQ Values to DataFrame: 
    totalq['BaseQ (cm/hr)'] = baseQ['Discharge (cm/hr)']
    
    #Return Variables: 
    return (baseQ, antQ_date, antQ_val, peakQ_date, peakQ, end_of_event, end_Q)

#%% Modified Time Series Plotting Containing Baseflow: 
    
#Define Function with Keyword Arguement for Baseflow: 
def timeseriesplot(df1, startdate, enddate, baseflow= None):    

    #Create Plot Area: 
    fig, ax1 = plt.subplots()

    #Plot Discharge Data: 
    ax1.plot(df1['Discharge (cm/hr)'], ',', linestyle='-', color='navy', label='Discharge (cm/hr)')

    #Axis Formatting: 
    ax1.set_ylim(bottom = 0)
    ax1.set_xlim(startdate, enddate)
    fig.autofmt_xdate()

    #Axis Labels: 
    ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
    ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
    
    #Optional Arguement Boolean Indicator: 
    if baseflow is not None: 
        ax1.plot(baseflow['Discharge (cm/hr)'], ',', linestyle='-', color='darkseagreen', 
                 label=' Baseflow (cm/hr)')
    
    #Legend: 
    fig.legend(bbox_to_anchor= (0.65, 0.0))   

#%% Running Functions per Watershed: 

#Albright: 
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_albright, watershed_area)
#Integrating Storm Total: 
storm_frame= df_albright[antQ_date : end_of_event]
albright_total= storm_frame['Discharge (cm/hr)'].sum()    
#Run Time Series Plotting Function: 
timeseriesplot(df_albright, df_albright.index[0], df_albright.index[-1], baseQ)

#Blackwater: 
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_blackwater, watershed_area)
#Integrating Storm Total: 
storm_frame= df_blackwater[antQ_date : end_of_event]
blackwater_total= storm_frame['Discharge (cm/hr)'].sum() 
#Run Time Series Plotting Function: 
timeseriesplot(df_blackwater, df_blackwater.index[0], df_blackwater.index[-1], baseQ)
    
#Bowden:
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_bowden, watershed_area)
#Integrating Storm Total: 
storm_frame= df_bowden[antQ_date : end_of_event]
bowden_total= storm_frame['Discharge (cm/hr)'].sum() 
#Run Time Series Plotting Function: 
timeseriesplot(df_bowden, df_bowden.index[0], df_bowden.index[-1], baseQ)
    
#Hendricks: 
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_hendricks, watershed_area)
#Integrating Storm Total: 
storm_frame= df_hendricks[antQ_date : end_of_event]
hendricks_total= storm_frame['Discharge (cm/hr)'].sum() 
#Run Time Series Plotting Function: 
timeseriesplot(df_hendricks, df_hendricks.index[0], df_hendricks.index[-1], baseQ)
    
#Parsons: 
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_parsons, watershed_area)
#Integrating Storm Total: 
storm_frame= df_parsons[antQ_date : end_of_event]
parsons_total= storm_frame['Discharge (cm/hr)'].sum() 
#Run Time Series Plotting Function: 
timeseriesplot(df_parsons, df_parsons.index[0], df_parsons.index[-1], baseQ)
    
#Rockville: 
#Run Hydrograph Seperation Function: 
hydrograph_sep(df_rockville, watershed_area)
#Integrating Storm Total: 
storm_frame= df_rockville[antQ_date : end_of_event]
rockville_total= storm_frame['Discharge (cm/hr)'].sum() 
#Run Time Series Plotting Function: 
timeseriesplot(df_rockville, df_rockville.index[0], df_rockville.index[-1], baseQ)
    
#%% Determine Effective Flow: 

#Define Function: 
def effect_flow(df): 
    
    #Calculate Effective Flow: 
    #df['Effective Flow (cm/hr)'] = df['Discharge (cm/hr)'] - df['BaseQ (cm/hr)']
    df['Eff Flow (cm/hr)']= np.where(df['Discharge (cm/hr)'] - df['BaseQ (cm/hr)'] > 0, 
                                     df['Discharge (cm/hr)'] - df['BaseQ (cm/hr)'], 0)
        
#Albright: 
effect_flow(df_albright)

#Blackwater: 
effect_flow(df_blackwater)

#Bowden: 
effect_flow(df_bowden)
    
#Hendricks: 
effect_flow(df_hendricks)

#Parsons: 
effect_flow(df_parsons)
    
#Rockville: 
effect_flow(df_rockville)

#%% Plotting Effective Flow Curves: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright['Eff Flow (cm/hr)'], ',', linestyle='-', color='navy', 
         label='Albright')
#Blackwater: 
ax1.plot(df_blackwater['Eff Flow (cm/hr)'], ',', linestyle='-', color='grey', 
         label='Blackwater')
#Bowden: 
ax1.plot(df_bowden['Eff Flow (cm/hr)'], ',', linestyle='-', color='dodgerblue', 
         label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks['Eff Flow (cm/hr)'], ',', linestyle='-', color='maroon', 
         label='Hendricks')
#Parsons: 
ax1.plot(df_parsons['Eff Flow (cm/hr)'], ',', linestyle='-', color='orange', 
         label='Parsons')
#Rockville: 
ax1.plot(df_rockville['Eff Flow (cm/hr)'], ',', linestyle='-', color='darkgreen', 
         label='Rockville')

#Axis Formatting: 
ax1.set_ylim(bottom = 0)
ax1.set_xlim(df_albright.index[0], df_albright.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Effective Flow Discharge Curves for Cheat River Watershed', 
             fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75))  

#%% Calculate Z-Score for Effective Flow:  

#Albright: 
df_albright['Z-Score EffQ']= (df_albright['Eff Flow (cm/hr)'] - 
                           df_albright['Eff Flow (cm/hr)'].mean()
                           ) / df_albright['Eff Flow (cm/hr)'].std()

#Blackwater: 
df_blackwater['Z-Score EffQ']= (df_blackwater['Eff Flow (cm/hr)'] - 
                           df_blackwater['Eff Flow (cm/hr)'].mean()
                           ) / df_blackwater['Eff Flow (cm/hr)'].std()

#Bowden: 
df_bowden['Z-Score EffQ']= (df_bowden['Eff Flow (cm/hr)'] - 
                           df_bowden['Eff Flow (cm/hr)'].mean()
                           ) / df_bowden['Eff Flow (cm/hr)'].std()

#Hendricks: 
df_hendricks['Z-Score EffQ']= (df_hendricks['Eff Flow (cm/hr)'] - 
                           df_hendricks['Eff Flow (cm/hr)'].mean()
                           ) / df_hendricks['Eff Flow (cm/hr)'].std()
    
#Parsons: 
df_parsons['Z-Score EffQ']= (df_parsons['Eff Flow (cm/hr)'] - 
                           df_parsons['Eff Flow (cm/hr)'].mean()
                           ) / df_parsons['Eff Flow (cm/hr)'].std()

#Rockville: 
df_rockville['Z-Score EffQ']= (df_rockville['Eff Flow (cm/hr)'] - 
                           df_rockville['Eff Flow (cm/hr)'].mean()
                           ) / df_rockville['Eff Flow (cm/hr)'].std()    

#%% Plotting Z-Scored Effective Flow Curves: 

#Create Plotting Area:     
fig, ax1 = plt.subplots()

#Plot Discharge Data: 
#Albright:
ax1.plot(df_albright['Z-Score EffQ'], ',', linestyle='-', color='navy', label='Albright')
#Blackwater: 
ax1.plot(df_blackwater['Z-Score EffQ'], ',', linestyle='-', color='grey', 
         label='Blackwater')
#Bowden: 
ax1.plot(df_bowden['Z-Score EffQ'], ',', linestyle='-', color='dodgerblue', label='Bowden')
#Hendricks: 
ax1.plot(df_hendricks['Z-Score EffQ'], ',', linestyle='-', color='maroon', 
         label='Hendricks')
#Parsons: 
ax1.plot(df_parsons['Z-Score EffQ'], ',', linestyle='-', color='orange', label='Parsons')
#Rockville: 
ax1.plot(df_rockville['Z-Score EffQ'], ',', linestyle='-', color='darkgreen', 
         label='Rockville')

#Axis Formatting: 
ax1.set_xlim(df_albright.index[0], df_albright.index[-1])
fig.autofmt_xdate()

#Axis Labels: 
ax1.set_ylabel('Discharge (cm/hr)', color='k', fontweight="bold", fontsize= 12)
ax1.set_xlabel('Date', color='k', fontweight="bold", fontsize= 12)
fig.suptitle('Z-Scored Effective Discharge Curves', 
             fontweight= "bold", fontsize=18)

#Legend: 
fig.legend(bbox_to_anchor= (1.15, 0.75))     

#%% Plotting Total Discharge Storm Event Volumes: 

#Create Plotting Area: 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

#Add Data Bars: 
locations = ['Albright', 'Davis', 'Bowden', 'Hendricks', 'Parsons', 'Rockville']
discharge_totals = [albright_total, blackwater_total, bowden_total, hendricks_total, 
            parsons_total, rockville_total]
ax.bar(locations, discharge_totals)

#Axis Labels: 
ax.set_ylabel('Total Discharge (cm)', color='k', fontweight="bold", fontsize= 12)
ax.set_xlabel('Location of Measurement', color='k', fontweight="bold", fontsize= 12)
ax.set_title('Total Storm Event Discharge Outputs', fontweight= "bold", fontsize=18)
    
#Display Bar Plot: 
plt.show()
    
#%% Pearson Coefficient Calculation for Time Series: 

#Albright-Davis Correlation: 
AlbrightDavisQ=df_albright['Discharge (cm/hr)'].corr(df_blackwater['Discharge (cm/hr)'])
    
#Albright-Bowden Correlation: 
AlbrightBowdenQ=df_albright['Discharge (cm/hr)'].corr(df_bowden['Discharge (cm/hr)'])
    
#Albright-Hendricks Correlation: 
AlbrightHendricksQ=df_albright['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
    
#Albright-Parsons Correlation:
AlbrightParsonsQ=df_albright['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
    
#Albright-Rockville Correlation: 
AlbrightRockvilleQ=df_albright['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])

#Davis-Bowden:
DavisBowdenQ=df_blackwater['Discharge (cm/hr)'].corr(df_bowden['Discharge (cm/hr)'])
    
#Davis-Hendricks:
DavisHendricksQ=df_blackwater['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
    
#Davis-Parsons:
DavisParsonsQ=df_blackwater['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])

#Davis-Rockville: 
DavisRockvilleQ=df_blackwater['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])

#Bowden-Hendricks: 
BowdenHendricksQ=df_bowden['Discharge (cm/hr)'].corr(df_hendricks['Discharge (cm/hr)'])
    
#Bowden-Parsons: 
BowdenParsonsQ=df_bowden['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
    
#Bowden-Rockville: 
BowdenRockvilleQ=df_bowden['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])

#Hendricks-Parsons: 
HendricksParsonsQ=df_hendricks['Discharge (cm/hr)'].corr(df_parsons['Discharge (cm/hr)'])
    
#Hendricks-Rockville: 
HendricksRockvilleQ=df_hendricks['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])

#Parsons-Rockville: 
ParsonsRockvilleQ=df_parsons['Discharge (cm/hr)'].corr(df_rockville['Discharge (cm/hr)'])

#%% End of Code
