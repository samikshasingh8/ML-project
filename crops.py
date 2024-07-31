#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# # LOADING THE DATASET

# In[5]:


df = pd.read_csv('Crop_recommendation.csv')


# # Encoding the target variable

# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])


# In[8]:


X = df.drop(['label', 'label_encoded'], axis=1)


# In[9]:


X


# In[10]:


y = df['label_encoded']


# In[11]:


y


# # Checking for missing values

# In[12]:


df.isna()


# # Splitting the dataset

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Standardinzing the scale

# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()


# # Training the KNN classifier

# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


knn = KNeighborsClassifier()


# # Selecting 10 best features by Recursive Feature Elimination(RFE)

# In[19]:


from sklearn.feature_selection import RFE


# In[20]:


rfe = RFE(knn, n_features_to_select=10)


# # Setting the pipeline

# In[21]:


from imblearn.pipeline import Pipeline


# In[22]:


pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('rfe', rfe),
    ('classifier', knn)
])


# # Performing grid search (best hyperparameters)

# In[23]:


from sklearn.model_selection import GridSearchCV


# In[24]:


param_grid = {
    'classifier__n_neighbors': [3, 5, 7],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}


# In[25]:


grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_macro')
grid_search.fit(X_train, y_train)


# In[26]:


best_model = grid_search.best_estimator_


# In[27]:


best_model


# In[28]:


y_pred = best_model.predict(X_test)


# In[29]:


y_pred


# # Evaluating the model

# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


accuracy_score(y_test,y_pred)


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


print(confusion_matrix(y_test, y_pred))


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


print(classification_report(y_test, y_pred))


# # NEW DATA PREDICTION

# In[41]:


new_data = {
    'N': [36],
    'P': [12],
    'K': [9],
    'temperature': [22],
    'humidity': [160],
    'ph': [6],
    'rainfall': [190]
}


# In[42]:


new_data_df = pd.DataFrame(new_data)


# In[43]:


new_data_df


# In[44]:


new_data_prediction = best_model.predict(new_data_df)


# In[45]:


predicted_crop = label_encoder.inverse_transform(new_data_prediction)


# In[46]:


print(predicted_crop)


# In[47]:


print(X.shape)  # Should output (number_of_samples, 5)


# In[48]:


import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('crop_recommendation_model.joblib')

# Dictionary for label-to-crop mapping
label_to_crop = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

# Title of the app with larger, bolder text
st.markdown(
    "<h1 style='text-align: center; font-size: 36px; font-weight: bold; color: #4CAF50;'>Crop Recommendation System</h1>",
    unsafe_allow_html=True
)

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

    try:
        prediction = model.predict(features)
        crop_name = label_to_crop.get(prediction[0], "Unknown crop")
        st.markdown(
            f"<h2 style='text-align: center; font-size: 30px; font-weight: bold; color: #FF5722;'>Recommended Crop: {crop_name}</h2>",
            unsafe_allow_html=True
        )

    except ValueError as e:
        st.error(f'Error: {e}')
