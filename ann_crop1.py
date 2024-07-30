#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# # LOADING THE DATASET

# In[6]:


df = pd.read_csv('Crop_recommendation.csv')


# # Encoding the target variable

# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])


# In[9]:


X = df.drop(['label', 'label_encoded'], axis=1)


# In[10]:


X


# In[11]:


y = df['label_encoded']


# In[12]:


y


# # Checking for missing values

# In[13]:


df.isna()


# # Splitting the dataset

# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Standardinzing the scale

# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()


# In[23]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Training the ANN

# In[18]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[19]:


model = Sequential()


# In[24]:


model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))



model.add(Dropout(0.3))





model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))


# In[26]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[27]:


history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)


# In[28]:


loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[ ]:





# # Selecting 10 best features by Recursive Feature Elimination(RFE)

# In[29]:


from sklearn.feature_selection import RFE


# In[31]:


rfe = RFE(model, n_features_to_select=10)


# # Setting the pipeline

# In[32]:


from imblearn.pipeline import Pipeline


# In[38]:


pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('rfe', rfe),
    ('classifier', model)
])


# # Performing grid search (best hyperparameters)

# no need in ann

# # Evaluating the model

# In[44]:


loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[45]:


from sklearn.metrics import confusion_matrix, classification_report


# In[46]:


y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)


# In[47]:


cm = confusion_matrix(y_test, y_pred_classes)


# In[49]:


print(cm)


# In[50]:


print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))


# In[51]:


model.save('crop_recommendation_ann.h5')


# # NEW DATA PREDICTION

# In[52]:


new_data = {
    'N': [36],
    'P': [12],
    'K': [9],
    'temperature': [22],
    'humidity': [160],
    'ph': [6],
    'rainfall': [190]
}


# In[53]:


new_data_df = pd.DataFrame(new_data)


# In[54]:


new_data_df


# In[58]:


new_data_scaled = scaler.transform(new_data_df)


# In[59]:


new_data_prediction = model.predict(new_data_scaled)


# In[60]:


predicted_class_index = np.argmax(new_data_prediction, axis=1)


# In[61]:


predicted_class_index = np.argmax(new_data_prediction, axis=1)


# In[62]:


predicted_crop = label_encoder.inverse_transform(predicted_class_index)


# In[63]:


print(predicted_crop)


# # Streamlit App Building

# In[69]:


import tensorflow as tf

model.save('crop_recommendation_ann.h5')


# In[70]:


import joblib

# Assuming `scaler` is your trained scaler
joblib.dump(scaler, 'scaler.joblib')


# In[72]:


import tensorflow as tf

# Load the ANN model
ann_model = tf.keras.models.load_model('crop_recommendation_ann.h5')


# In[73]:


import joblib

# Load the scaler
scaler = joblib.load('scaler.joblib')


# In[74]:


import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# Load the ANN model and scaler
ann_model = tf.keras.models.load_model('crop_recommendation_ann.h5')
scaler = joblib.load('scaler.joblib')

# Dictionary for label-to-crop mapping
label_to_crop = {index: label for index, label in enumerate(label_encoder.classes_)}

# Title of the app with color
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Crop Recommendation System</h1>", unsafe_allow_html=True)

# Collecting user inputs with icons and colors
st.markdown("<h2 style='color: #2196F3;'>Enter the features below:</h2>", unsafe_allow_html=True)

N = st.number_input('🌿 Nitrogen content (N)', min_value=0, max_value=100, value=50, step=1)
P = st.number_input('🌾 Phosphorus content (P)', min_value=0, max_value=100, value=50, step=1)
K = st.number_input('🌱 Potassium content (K)', min_value=0, max_value=100, value=50, step=1)
pH = st.number_input('🧪 pH of soil', min_value=1.0, max_value=14.0, value=7.0, step=0.1)
humidity = st.number_input('💧 Humidity (%)', min_value=0, max_value=100, value=50, step=1)
rainfall = st.number_input('🌧️ Rainfall (mm)', min_value=0, max_value=500, value=250, step=1)
temperature = st.number_input('🌡️ Temperature (°C)', min_value=-10, max_value=50, value=25, step=1)

# Predict button with color
if st.button('🔮 Predict my crop', key='predict'):
    features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    features_scaled = scaler.transform(features)

    try:
        # Use ANN model for prediction
        prediction = ann_model.predict(features_scaled)
        prediction = np.argmax(prediction, axis=1)

        crop_name = label_to_crop.get(prediction[0], "Unknown crop")
        st.markdown(
            f"<h2 style='text-align: center; color: #FF5722;'>Recommended Crop: {crop_name}</h2>",
            unsafe_allow_html=True
        )

    except ValueError as e:
        st.error(f'Error: {e}')
