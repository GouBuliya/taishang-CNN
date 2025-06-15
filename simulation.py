#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:00:00 2024

@author: hakan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import talib
import mplfinance as mpf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img
import tensorflow as tf
import shutil
import json


# 加载配置文件
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

gpus = tf.config.list_physical_devices('GPU')
# if gpus: 
#     for gpu in gpus:
#           tf.config.experimental.set_memory_growth(gpu, True)
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=config['gpu']['memory_limit_mb'])]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

dd = config['data']['test_data_file']
initialTime_index = config['time_window']['initial_time_index']
finalTime_index = config['time_window']['final_time_index']
#this part creates images without labels. Because we predict the labels using our cnn model. 

data = pd.read_csv(dd, delimiter=config['data']['delimiter'], index_col=config['data']['time_column'], parse_dates=True)
data['SMA'] = talib.SMA(data['Close'], timeperiod=config['technical_indicators']['sma_period'])
data=data[initialTime_index:finalTime_index]

output_dir = config['image_processing']['output_dir']
shutil.rmtree(output_dir,ignore_errors=True)

os.makedirs(output_dir, exist_ok=True)
window_size = config['time_window']['window_size']
shift_size = config['time_window']['shift_size']
for i in range(0, len(data) - window_size, shift_size):
    window = data.iloc[i:i+window_size]
    save_path = os.path.join(output_dir, f"{window.iloc[-1].name}.png")
    ap = [mpf.make_addplot(window['SMA'], color=config['technical_indicators']['sma_color'], secondary_y=False)]
    mpf.plot(window, type=config['chart_plotting']['chart_type'], style=config['chart_plotting']['chart_style'], 
             addplot=ap, volume=config['chart_plotting']['volume'], axisoff=config['chart_plotting']['axisoff'], 
             ylabel=config['chart_plotting']['ylabel'], savefig=save_path)
    plt.close()

# Create DataFrame
data = pd.read_csv(dd, delimiter=config['data']['delimiter'], parse_dates=True)
data=data[initialTime_index:finalTime_index]
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df[config['data']['time_column']])
df['Date'] = df['Date'].map(mdates.date2num)  # Convert dates to matplotlib format

# Create subplots and plot candlestick chart
fig, ax = plt.subplots(figsize=config['chart_plotting']['figure_size'])
candlestick_ohlc(ax, df[['Date', 'Open', 'High', 'Low', 'Close']].values, 
                 width=config['chart_plotting']['candle_width'], 
                 colorup=config['chart_plotting']['up_color'], 
                 colordown=config['chart_plotting']['down_color'])


# This part makes predictions 
from tensorflow.keras.utils import img_to_array

dataset_path = config['image_processing']['output_dir']
X=[]
for name in os.listdir(dataset_path):
    image1 = load_img(dataset_path  + '/' + name, 
                     color_mode=config['image_processing']['color_mode'], 
                     interpolation=config['image_processing']['interpolation'],
                     target_size=tuple(config['image_processing']['target_size']))  # MODEL 2 & MODEL 3 (analyzing each image as a whole)
    image1 = img_to_array(image1)
    image1 = image1 / 255
    X.append(image1)
X=np.array(X) 
# model=load_model("chart_classification_model.h5")
model=load_model(config['model']['model_file'])
predictions = model.predict(X)

image_names=os.listdir(dataset_path)
indicator_xcoordinates=[]
indicator_trends=[]
for idx,i in enumerate(predictions):
    if i>=config['model']['prediction_threshold']:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append(config['trading']['buy_signal'])
    else:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append(config['trading']['sell_signal'])

## remove consecutive the same signals. That is the list will be [up,down,up,down,up, down...] and so on
signal_x = [indicator_xcoordinates[0]]  # Start with the first element
signal_label = [indicator_trends[0]] 
for i in range(1, len(indicator_trends)):
    if indicator_trends[i] != indicator_trends[i - 1]:
        signal_x.append(indicator_xcoordinates[i])
        signal_label.append(indicator_trends[i])
        
indicator_xcoordinates=signal_x
indicator_trends=signal_label  
   
# Add annotations for up/down labels
for time, label in zip(indicator_xcoordinates, indicator_trends):
    # Get the row data for that specific time
    if time in df[config['data']['time_column']].values:
        result = df.isin([time])
        locations = result.stack()[result.stack()] 
        row = locations.index[0][0]
        row=df.loc[row]
        timestamp = mdates.date2num(pd.to_datetime(time))
        # Position label above the high price with a small offset
        if label==config['trading']['sell_signal']:
            y_position = row['High'] + config['signal_annotation']['down_offset']  # Adjust offset as needed
        else:
            y_position = row['Low'] + config['signal_annotation']['up_offset']  # Adjust offset as needed
        ax.annotate(label,
                    xy=(timestamp, y_position),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha=config['signal_annotation']['horizontal_alignment'],
                    va=config['signal_annotation']['vertical_alignment'],
                    fontsize=config['signal_annotation']['fontsize'],
                    bbox=dict(boxstyle=config['signal_annotation']['box_style'], 
                             fc=config['signal_annotation']['face_color'], 
                             alpha=config['signal_annotation']['alpha'])
                    )
    else:
            print(f"Time {time} not found in data")
            print(time)
# Formatting the plot
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter(config['chart_plotting']['date_format']))
plt.title(config['chart_plotting']['title'])
plt.xticks(rotation=config['chart_plotting']['rotation'])
plt.xlabel(config['chart_plotting']['xlabel'])
plt.ylabel(config['chart_plotting']['ylabel_final'])
plt.grid(config['chart_plotting']['grid'])
plt.tight_layout()
plt.show()


initial_amount_usd = config['trading']['initial_amount_usd']  # Initial amount in USD
current_amount_usd = initial_amount_usd
amount_in_euros = 0  # Amount of euros after buying
number_changes=0
for i in range(len(indicator_xcoordinates)):
    time=indicator_xcoordinates[i]
    if time in df[config['data']['time_column']].values:
        result = df.isin([time])
        locations = result.stack()[result.stack()] 
        row = locations.index[0][0]
        row=df.loc[row]
    if indicator_trends[i] == config['trading']['buy_signal'] and current_amount_usd > 0:  # Buy signal   
        amount_in_euros = current_amount_usd / row['Open']
        number_changes+=1
        current_amount_usd = 0  # All money converted to euros
        print(f"Bought at {time} at price {row['Open']}, amount in euros: {amount_in_euros:.2f}")
        
    elif indicator_trends[i] == config['trading']['sell_signal'] and amount_in_euros > 0:  # Sell signal
        current_amount_usd = amount_in_euros * row['Open']
        amount_in_euros = 0  # Euros converted back to USD
        print(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}")
        number_changes+=1

# Final amount
if amount_in_euros > 0:  # Convert any remaining euros to USD at the last close price
    print(f"\nFinal amount in EUR: {amount_in_euros:.2f}")
    current_amount_usd=amount_in_euros*data['Open'][finalTime_index-1]
    print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
else:
    print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
    
print(f"\nTotal number of buy/sell: {number_changes}")