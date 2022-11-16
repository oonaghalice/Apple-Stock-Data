#Import necessary functions- this is toms code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab 
import scipy.stats as stats
import math
import matplotlib.dates as mdates

#------------------------------------------------------------------------------

#Read in Data and Save Close price to an array
data= pd.read_csv(r'/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPL.csv',
                  usecols=['Adj Close'])
data=np.array(data)
print(len(data))
Cl=np.zeros(len(data))
for i in range(len(data)):
    Cl[i]=data[i][0]

print('Closing Price:',Cl)
print('Total Days:' ,len(Cl))

#Store all the dates 
columns = ["Date","Open","High","Low","Close","Adj Close","Volume"]
full = pd.read_csv(r"/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPL.csv", usecols=columns)
Date = full["Date"]
Adj = full["Adj Close"]

#------------------------------------------------------------------------------

#Plot of the apple stock price over the last 9 years
Date = mdates.date2num(Date)
plt.figure(figsize=(10,5))
plt.plot_date(Date,Adj,markersize=0.2,linestyle='solid', color='b')
plt.title('Apple Stock Price', fontsize=18)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Price in Dollars', fontsize=15)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLStockPrice.png')
plt.show()

#Array storing the annual returns of the stock
anret = [-7.277857, 3.820356, 12.979643, -3.665001, 8.4475, 5.360443, 6.5075,
         16.5125, 58.200005, 51.940002]
years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

plt.plot(years, anret, marker='*', color='b')
plt.title('Plot of Annual Returns', fontsize=18)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Annual Return', fontsize=15)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLAnnualReturn.png')
plt.show()

#------------------------------------------------------------------------------

#Splitting the data into the financial years (5th april to 5th next year)
ret12 = Cl[:251]
ret13 = Cl[252:503]
ret14 = Cl[504:754]
ret15 = Cl[755:1006]
ret16 = Cl[1007:1259]
ret17 = Cl[1260:1510]
ret18 = Cl[1511:1762]
ret19 = Cl[1763:2013]
ret20 = Cl[2014:2264]
ret21 = Cl[2265:]

#------------------------------------------------------------------------------

#Calculating daily returns
Ret=np.zeros(len(Cl)-1)
for i in range(len(Cl)-1):
    Ret[i]=100*(Cl[i+1]-Cl[i])/Cl[i]

#Plotting Daily return as a histogram - PART1A   
plt.hist(Ret, bins=60, rwidth=0.5, color='b')
plt.title('Histogram of Daily Returns', fontsize=18)
plt.xlabel('Daily Return', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
maxi = np.where(Ret == Ret.max())
print('Peak of the Histogram:', maxi)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLHistogram.png')
plt.show()

#QQ-Plot of returns vs normal ditribution - PART1B
stats.probplot(Ret, dist="norm", plot=pylab)
plt.title('Normal Quantile-Quantile Plot', fontsize=18)
plt.xlabel('Theoretical Quantiles', fontsize=15)
plt.ylabel('Ordered Values', fontsize=15)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLQQPlot.png')
pylab.show()

#------------------------------------------------------------------------------

#Calculating the drift and volatility - PART 1C
logret12 = []
trash12 = []
for i in range(len(Cl)):
    for j in range(len(Cl)-1):
        x = Cl[j]
        y = Cl[i]
    lnr = np.log(x/y)
    if lnr == 0:
        trash12.append(lnr)
    if math.isnan(lnr) == True:
        trash12.append(lnr)
    else:
        logret12.append(lnr) 
#The drift is the average log-return value x no of trading days       
drift = np.mean(logret12)*252
vol=np.std(Ret)*(len(ret18))**0.5

print('Drift:', drift)
print('Volatility:', vol)

#------------------------------------------------------------------------------

#PART 1D- predicting the stock price
#Tutorial 4

#Next Quarter
s_0=Cl[2515]
Smax=s_0+((drift/100)*s_0*(0.25))+((vol/100)*s_0*(0.25**0.5)*2)
Smin=s_0+((drift/100)*s_0*(0.25))-((vol/100)*s_0*(0.25**0.5)*2)
print('The minimum stock price next quater with a 95% confidence level is:', Smin)
print('The maximum stock price next quater with a 95% confidence level is:', Smax)

#Next 6 Months
Smax6=s_0+((drift/100)*s_0*(0.5))+((vol/100)*s_0*(0.5**0.5)*2)
Smin6=s_0+((drift/100)*s_0*(0.5))-((vol/100)*s_0*(0.5**0.5)*2)
print('The minimum stock price in 6 months with a 95% confidence level is:', Smin6)
print('The maximum stock price in 6 months with a 95% confidence level is:', Smax6)

#Next Year
Smax1=s_0+((drift/100)*s_0*(1))+((vol/100)*s_0*(1**0.5)*2)
Smin1=s_0+((drift/100)*s_0*(1))-((vol/100)*s_0*(1**0.5)*2)
print('The minimum stock price for next year with a 95% confidence level is:', Smin1)
print('The maximum stock price for next year with a 95% confidence level is:', Smax1)

#------------------------------------------------------------------------------

#PART 2 OF THE CW

#Calculate Upward and downward trend 
u=np.exp((vol/100)*(1/12)**0.5)
d=np.exp(-(vol/100)*(1/12)**0.5)
print('Upward trend: ',u)
print('Downward trend: ',d)

#Probabilities of movement
#Assuming 0.00814 is the LIBOR interest rate over the 10 years
r=0.00814
pu=(np.exp(0.00814*(1/12))-d)/(u-d)
prd=1-pu
print('Probaility of upward move: ',pu)
print('Probaility of downward move: ',prd)

#strike price for call and put
stcall=185
stput=170


#pa is the last share price from the data
pa=s_0
pb=pa*u
pc=pa*d
pd=pb*u
pe=pc*u
pf=pc*d

print('Share price at node A: ',pa)
print('Share price at node B: ',pb)
print('Share price at node C: ',pc)
print('Share price at node D: ',pd)
print('Share price at node E: ',pe)
print('Share price at node F: ',pf)

#Long Put pay off
#An investor has bought the option to SELL a stock to the writer in the future
Pd=max(stput-pd,0)
Pe=max(stput-pe,0)
Pf=max(stput-pf,0)
print('Long put pay off at node D: ',Pd)
print('Long put pay off at node E: ',Pe)
print('Long put pay off at node F: ',Pf)

#Put option pricing
Pb=np.exp(-0.00814*(1/12))*(pu*Pd+prd*Pe)
Pc=np.exp(-0.00814*(1/12))*(pu*Pe+prd*Pf)
Pa=np.exp(-0.00814*(1/12))*(pu*Pb+prd*Pc)
print('Put option fair price at node A: ',Pa)
print('Put option fair price at node B: ',Pb)
print('Put option fair price at node C: ',Pc)

#Long Call pay off
#An investor has bought the option to BUY a stock to the writer in the future
Pcd=max(pd-stcall,0)
Pce=max(pe-stcall,0)
Pcf=max(pf-stcall,0)
print('Long call pay off at node D: ',Pcd)
print('Long call pay off at node E: ',Pce)
print('Long call pay off at node F: ',Pcf)

#Call option price
Pcb=np.exp(-0.00814*(1/12)) *(pu*Pcd+prd*Pce)
Pcc=np.exp(-0.00814*(1/12)) *(pu*Pce+prd*Pcf)
Pca=np.exp(-0.00814*(1/12)) *(pu*Pcb+prd*Pcc)
print('Call option fair price at node A: ',Pca)
print('Call option fair price at node B: ',Pcb)
print('Call option fair price at node C: ',Pcc)

#------------------------------------------------------------------------------

#Profit/loss diagram

xcall = [178.44, 185, 210.26]
callpayoff = [0, 0, 25.26]
xput = [151.43, 170, 178.44]
putpayoff = [18.56, 0, 0]
callprofloss = [-5.9, -5.9, 19.36]
putprofloss = [13.62, -4.94, -4.94]


plt.plot(xcall, callpayoff, color='r', label='Call Option Payoff')
plt.plot(xcall ,callprofloss, color='b', label='Call Profit/Loss')
plt.title('Profit/Loss Diagram for European Call Option', fontsize=18)
plt.xlabel('Share Price ($)', fontsize=15)
plt.ylabel('Payoff and Profit/Loss ($)', fontsize=15)
plt.axvline(x=185, color='black', linestyle='-.', label='Strike Price')
plt.axhline(y=0, color='black', linestyle=':')
plt.legend(fontsize=10)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLprofitcall.png')
plt.show()

plt.plot(xput, putpayoff, color='r', label='Put Option Payoff')
plt.plot(xput ,putprofloss, color='b', label='Put Profit/Loss')
plt.title('Profit/Loss Diagram for European Put Option', fontsize=18)
plt.xlabel('Share Price ($)', fontsize=15)
plt.ylabel('Payoff and Profit/Loss ($)', fontsize=15)
plt.axvline(x=170, color='black', linestyle='-.',label='Strike Price')
plt.axhline(y=0, color='black', linestyle=':')
plt.legend(fontsize=10)
plt.savefig('/Users/oonaghparker/Desktop/YR3 SEM2 CW/Financial Derivatives/AAPLprofitput.png')
plt.show()













