



%matplotlib inline
import matplotlib.pyplot as plt
from IPython.html.widgets import *
import numpy as np


def pltsin(f):
    x=np.linspace(0,1,100)
    plt.plot(x,np.sin(2*np.pi*x*f));
    plt.show()
  

interact(pltsin, f=(1,10,0.1))






from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from netCDF4 import Dataset, num2date, date2num

import pandas as pd

N_samples = 25
x_min = -5
x_max = 5
x1= np.linspace(x_min,x_max,N_samples*5)
x= np.random.choice(x1,size=N_samples)
noise_std=1
noise_mean=0
noise_magnitude = 2

def func_gen(N_samples,x_min,x_max,noise_magnitude,noise_sd,noise_mean):
    x1= np.linspace(x_min,x_max,N_samples*5)
    x= np.random.choice(x1,size=N_samples)
    y=2*x-0.6*x**2+0.2*x**3+18*np.sin(x)
    y1=2*x1-0.6*x1**2+0.2*x1**3+18*np.sin(x1)
    y= y+noise_magnitude*np.random.normal(loc=noise_mean,scale=noise_sd,size=N_samples)
    plt.figure(figsize=(8,5))
    plt.plot(x1,y1,c='k',lw=2)
    plt.scatter(x,y,edgecolors='k',c='yellow',s=60)
    plt.grid(True)
    plt.show()
    return (x,y,x1,y1)



p=interactive(func_gen,N_samples={'Low (50 samples)':50,'High (200 samples)':200},x_min=(-5,0,1), x_max=(0,5,1),
              noise_magnitude=(0,5,1),noise_sd=(0.1,1,0.1),noise_mean=(-2,2,0.5))
display(p)







variabs = ['air', 'uwnd', 'vwnd', 'rhum']
for vvv in variabs:
    for i in range(2000,2019):
        !wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/{vvv}.sig995.{i}.nc

air = f.variables['air']

plt.imshow(air[0,:,:])

def sh(time):
    plt.imshow(air[time,:,:])
    
    
sh(100)


interact(sh, time=(0,355,1));

def sh(var='air', time=0):
    f = Dataset(var+'.sig995.2000.nc')
    vv = f.variables[var]
    plt.imshow(vv[time,:,:])
    
variabs = ['air', 'uwnd', 'vwnd', 'rhum']

interact(sh, time=(0,355,1), var=variabs);

def sh(year='2000',var='air', time=0):
    f = Dataset(var+'.sig995.'+year+'.nc')
    vv = f.variables[var]
    plt.imshow(vv[time,:,:])
    
years = [str(x) for x in range(2000,2019)]

interact(sh, year=years, time=(0,355,1), var=variabs);


##############


import cartopy.crs as ccrs
import time as tme
import matplotlib.pyplot as plt


#m = Basemap(projection='npstere',boundinglat=60,lon_0=0,resolution='l')

lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
lon, lat = np.meshgrid(lon, lat)


def sh(year=2000,var='air', FixedColor=False, vm=-3., vma=0., inte=10, time=0):
    f = Dataset(var+'.sig995.'+str(year)+'.nc')
    vv = f.variables[var]
    tt = f.variables['time']
    dd = num2date(tt[time], tt.units)

    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.contourf(lon,lat,vv[time,:,:]-273.15, transform=ccrs.PlateCarree())

    ax.coastlines()

    plt.colorbar()
    plt.title(str(dd))

    plt.show()

        
  

#for current_year in [2000,2018]:
#    for current_time in [0,354]:
#        #print (str(current_year) + " " + str(current_time)
#        sh (year=current_year, time=current_time)
#        tme.sleep(4)
        
        
interact(sh, year=(2000,2019,1), time=(0,355,1), var=variabs,\
         vm=FloatText(value=250), vma=FloatText(value=300), inte=FloatText(value=10));

############

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

plt.show()



#####

import ipysheet
sheet = ipysheet.sheet()
sheet

sheet = ipysheet.sheet(rows=3, columns=4)
cell1 = ipysheet.cell(0, 0, 'Hello')
cell2 = ipysheet.cell(2, 0, 'World')
cell_value = ipysheet.cell(2,2, 42.)
sheet


########


import ipywidgets as widgets
sheet = ipysheet.sheet(rows=3, columns=2, column_headers=False, row_headers=False)
cell_a = ipysheet.cell(0, 1, 1, label_left='a')
cell_b = ipysheet.cell(1, 1, 2, label_left='b')
cell_sum = ipysheet.cell(2, 1, 3, label_left='sum', read_only=True)

# create a slider linked to cell a
slider = widgets.FloatSlider(min=-10, max=10, description='a')
widgets.jslink((cell_a, 'value'), (slider, 'value'))

# changes in a or b should trigger this function
def calculate(change):
    cell_sum.value = cell_a.value + cell_b.value

cell_a.observe(calculate, 'value')
cell_b.observe(calculate, 'value')


widgets.VBox([sheet, slider])



##########


