<h1>LSTM Weather Forecasting</h1>

### [Medium Article Demonstration](https://medium.com/@sebastienwebdev/forecasting-weather-patterns-with-lstm-a-python-guide-without-dates-433f0356136c)

<h2>Description</h2>
<p>This project demonstrates using LSTM networks for weather pattern forecasting, focusing on temperature predictions without the need for explicit date columns. It uses the "Indian Weather Repository" dataset from Kaggle, highlighting the potential of LSTM in predictive modeling.</p>

<h2>Environment & Utilities Used</h2>
<ul>
  <li><b>Python</b></li>
  <li><b>NumPy, Pandas</b></li>
  <li><b>Matplotlib, Seaborn</b></li>
  <li><b>TensorFlow</b></li>
  <li><b>Scikit-learn</b></li>
</ul>

<h2>Program Walk-through:</h2>

<h3>Data Preparation</h3>
<pre><code>
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
</code></pre>
<h3>Load and normalize the dataset</h3>
<pre><code>df = pd.read_csv('path/to/IndianWeatherRepository.csv')
data = df[['temperature']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
</code></pre>

<h3>LSTM Model Training</h3>
<pre><code>
# Define and compile the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(None, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
</code></pre>
<h3>Train the model</h3>
<pre><code>model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1)
</code></pre>

<h3>Results Visualization</h3>
<pre><code>
# Plotting predicted vs actual temperatures
plt.figure(figsize=(10,6))
plt.plot(actual_temperature, label='Actual Temperature')
plt.plot(predicted_temperature, label='Predicted Temperature', linestyle='--')
plt.title('Temperature Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()
</code></pre>

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*T97Ont-AikhUzIBbhKuJCw.png" alt="Temperature Prediction Graph" />

<h2>Conclusion</h2>
<p>The use of LSTM for predicting weather patterns shows significant potential even without specific date data, demonstrating the model's ability to understand and forecast based on sequential data alone.</p>
