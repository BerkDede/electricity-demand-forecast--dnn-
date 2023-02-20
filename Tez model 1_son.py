# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:08:10 2021

@author: Berk Dede
"""




import matplotlib.pyplot as plt
import pandas as pd
import math
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error



df_1 = pd.read_csv('tez3.csv')
df_2 = pd.read_csv('time_stamp.csv')
df_datetime=df_1.iloc[:,0]
df3=df_datetime.str.split(expand=True)
df_date=df3.iloc[:,0]
df_hour=df3.iloc[:,1]
df_hour=df_hour.str.split(pat=':',expand=True)
df_date=df_date.str.split(pat='/',expand=True)
day=df_date.iloc[:,1:2]
day.set_axis(['day'],axis=1,inplace=True)
day=day.astype(int)
day=((day)%7)
df_lag=df_1.iloc[:,1:2].shift(24).fillna(0)
df_lag.rename(columns={'consumption': 'consumption_lag'}, inplace=True)


mon=df_date.iloc[:,0:1]
mon.set_axis(['mounth'],axis=1,inplace=True)
mon=mon.astype(int)
year=df_date.iloc[:,2:3]
year.set_axis(['year'],axis=1,inplace=True)
year=year.astype(int)
hour=df_hour.iloc[:,0:1]
hour.set_axis(['hour'],axis=1,inplace=True)
hour=hour.astype(int)
frames = [df_1, df_2,df_lag,year,day,mon,hour,]
df= pd.concat(frames,axis=1)


df=df.drop('time_stamp',axis=1)


#Raw Data Visualization

titles = [
 
    "MWh",
    "mm / hour",
    "°C",
    "W / m²",
    "W / m²",
    "mm / hour",
    "kg / m²",
    "[0,1]",
    "kg / m³",
    "MWh",
    
]

feature_keys = [
    
    "consumption",
    "precipitation",
    "temperature",
    "irradiance_surface",
    "irradiance_toa",
    "snowfall",
    "snow_mass",
    "cloud_cover",
    "air_density",
    "consumption_lag"
    
    
   
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "darkblue"
    
    ]

date_time_key = "Date_time"

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=5, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


show_raw_visualization(df)


plt.matshow(df.corr())
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.gca().xaxis.tick_bottom()
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

#date time to index
df2=df.set_index(["Date_time"])
#ensure all data is float
values = df2.values.astype('float32')



X = values[:,1:15]

Y = values[:,0:1]

#The output is a large table of statistics results. Note that we are interested only in the P-value result
#P value sholdn't higher than significance level
 
model1 = sm.OLS (Y,X).fit()
print(model1.summary())

remove = ['snowfall']
values2= df2[df2.columns.difference(remove)].values.astype('float32')

X2=values2[:,1:14]
model3 = sm.OLS (Y,X2).fit()
print(model3.summary())




# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler.fit_transform(X2)
scaled_Y = scaler.fit_transform(Y)


X_train= scaled_X [:34728,:]
X_test= scaled_X [34728:,:]
y_train = scaled_Y [:34728,:]
y_test = scaled_Y [34728:,:]
X_test_1h= scaled_X [34728:34896,:]
y_test_1h = scaled_Y [34728:34896,:]

time= df.iloc[:,0:1]
time_train = time.iloc[:34728,:]
time_test= df.iloc[34728:,0:1]

time_test2=time_test.squeeze()
"""
time_test2=time_test2.to_numpy()
"""
time2=df2.iloc[34728:,0:1]
time3=df2.iloc[34728:34896,0:1]

model= Sequential()
model.add(Dense(12,kernel_initializer='he_normal',activation='relu'))

model.add(Dense(50,kernel_initializer='he_normal',activation='relu'))

model.add(Dense(50,kernel_initializer='he_normal',activation='relu'))

model.add(Dense(50,kernel_initializer='he_normal',activation='relu'))

model.add(Dense(50,kernel_initializer='he_normal',activation='relu'))



model.add(Dense(1, activation='linear'))
model.compile(optimizer = 'adam', loss = 'mae', metrics=["mae","mape","mse"])

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, 
                                            patience=15)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(X_train,y_train,
                validation_split=0.2,
                batch_size=1024, 
                shuffle=True,
                verbose=2,
                epochs=500,
                callbacks=[es_callback, modelckpt_callback])

model.summary()
print(path_checkpoint)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'vall'], loc='upper right')
plt.show()

Predict = model.predict(X_test)
Predict_inv=scaler.inverse_transform(Predict)
y_test_inv=scaler.inverse_transform(y_test)

Predict_data=pd.DataFrame(Predict_inv)
Test_data=pd.DataFrame(y_test_inv)

frames2= [Predict_data,Test_data]
Test=pd.concat(frames2,axis=1)
Test2=Test.astype('float')
Fark=(Test_data)-(Predict_data)
Fark2=pd.DataFrame(Fark)
test_mse = math.sqrt(mean_squared_error(Predict_data,Test_data))
test_mae=mean_absolute_error(Predict_data,Test_data)
test_mape=mean_absolute_percentage_error(Predict_data,Test_data)
frame_test=[test_mse,test_mae,test_mape]


                     
Predict_data2=Predict_data.astype('float64')
Test_data2=Test_data.astype('float64')
index_test=pd.DataFrame(index=time2.index)
frames3= [Predict_data,Test_data]
Test_plot=pd.concat(frames3,axis=1)
Test_plot=Test_plot.astype('float')
Test_plot2 = pd.DataFrame(Test_plot.values,index=time2.index)
Test_plot2.columns =('Predict', 'Test')

plt.figure(figsize=(20,4))
plt.plot(Test_plot2)
plt.xticks(time2.index[::20],fontsize=10,rotation=60)
plt.ylabel('MWh')
plt.legend(['Pred', 'Real'], loc='upper right')
plt.show()

print(frame_test)

Predict_1h = model.predict(X_test_1h)
Predict_inv_1h=scaler.inverse_transform(Predict_1h)
y_test_inv_1h=scaler.inverse_transform(y_test_1h)
Predict_data_1h=pd.DataFrame(Predict_inv_1h)
Test_data_1h=pd.DataFrame(y_test_inv_1h)


Predict_data_1h=Predict_data_1h.astype('float64')
Test_data_1h=Test_data_1h.astype('float64')
index_test=pd.DataFrame(index=time2.index)
frames_1h= [Predict_data_1h,Test_data_1h]
Test_plot=pd.concat(frames_1h,axis=1)
Test_plot_1h=Test_plot.astype('float')
Test_plot_1h_2 = pd.DataFrame(Test_plot_1h.values,index=time3.index)
Test_plot_1h_2.columns =('Predict', 'Test')

plt.figure(figsize=(20,4))
plt.plot(Test_plot_1h_2)
plt.xticks(time3.index[::20],fontsize=10,rotation=60)
plt.ylabel('MWh')
plt.legend(['Pred', 'Real'], loc='upper right')
plt.show()

test_mse_2 = math.sqrt(mean_squared_error(Predict_data_1h,Test_data_1h))
test_mae_2=mean_absolute_error(Predict_data_1h,Test_data_1h)
test_mape_2=mean_absolute_percentage_error(Predict_data_1h,Test_data_1h)
frame_test2=[test_mse_2,test_mae_2,test_mape_2]
print(frame_test2)