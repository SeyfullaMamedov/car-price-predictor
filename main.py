from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from home import home_page
from us import about_us

# Read the normalized dataframe
df_normalized = pd.read_csv('car_prediction6.csv')

# Read the original dataframe
df_original = pd.read_csv('final_data7.csv')

# Map color names to numeric codes
color_map = {'Altın': 0, 'Bej': 1, 'Beyaz': 2, 'Bordo': 3, 'Diğer': 4, 'Füme': 5, 'Gri': 6, 'Gri (Gümüş)': 7,
             'Gri (metalik)': 8, 'Gri (titanyum)': 9, 'Kahverengi': 10, 'Kırmızı': 11, 'Lacivert': 12, 'Mavi': 13,
             'Mavi (metalik)': 14, 'Mor': 15, 'Pembe': 16, 'Sarı': 17, 'Siyah': 18, 'Turkuaz': 19, 'Turuncu': 20,
             'Yeşil': 21, 'Yeşil (metalik)': 22, 'Şampanya': 23}

# Map brand names to numeric codes
brand_map = {'Audi': 0, 'Citroen': 1, 'Fiat': 2, 'Ford': 3, 'Opel': 4, 'Renault': 5, 'Toyota': 6, 'Volkswagen': 7}

model_map = {
    '100': 0,
    '80': 1,
    'A1': 2,
    'A3': 3,
    'A4': 4,
    'A5': 5,
    'A6': 6,
    'A8': 7,
    'Astra': 8,
    'BX': 9,
    'C-Elysee': 10,
    'C1': 11,
    'C2': 12,
    'C3': 13,
    'C4': 14,
    'C5': 15,
    'Clio': 16,
    'Corolla': 17,
    'Corsa': 18,
    'Egea': 19,
    'Evasion': 20,
    'Fiesta': 21,
    'Fluence': 22,
    'Focus': 23,
    'Golf': 24,
    'Linea': 25,
    'Polo': 26,
    'Saxo': 27,
    'TT': 28,
    'Xsara': 29,
    'ZX': 30
}


engine_options = [0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.33, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

df_normalized['Color'] = df_original['Color'].map(color_map)
df_normalized['Car Brand'] = df_original['Car Brand'].map(brand_map)
df_normalized['Model'] = df_original['Model'].map(model_map)

# Ensure that all features are properly encoded
X = df_normalized.drop(['Price', 'Color'], axis=1)
y = df_normalized['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_params = {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100}
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 10, 20],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Perform GridSearchCV to find the best parameters
# rf = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
#
# # Get the best parameters from the grid search
# best_params = grid_search.best_params_
# st.write("Best Parameters from GridSearchCV:", best_params)
# rf = RandomForestRegressor()
# rf.fit(X_train, y_train)
# y_pred_test = rf.predict(X_test)
# r2_test = r2_score(y_test, y_pred_test)
# print("Test R^2 Score:", r2_test)
#
# # Perform cross-validation
# cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
# print("Cross-validation R^2 scores:", cv_scores)
# print("Mean Cross-validation R^2 score:", cv_scores.mean())
# print("Standard deviation of Cross-validation R^2 scores:", cv_scores.std())

CUSTOM_CSS = """
<style>
    /* Change background color */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Change font style and color */
    h1, h2, h3, h4, h5, h6, p, .stTextInput>div>div>input, .stSelectbox>div>div>select {
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
        color: #333;
    }
    /* Adjust specific elements if needed */
</style>
"""

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def load_image(image_path, width=None):
    img = Image.open('CN.png')
    if width:
        st.image(img, width=width)
    else:
        st.image(img)

current_year = datetime.now().year

brand_models = {
    'Audi': ['80', '100', 'A1', 'A3', 'A4', 'A5', 'A6', 'A8', 'TT'],
    'Citroen': ['BX', 'C-Elysee', 'C1', 'C2', 'C3', 'C4', 'C5', 'Evasion', 'Saxo', 'Xsara', 'ZX'],
    'Fiat': ['Linea', 'Egea'],
    'Ford': ['Fiesta', 'Focus'],
    'Opel': ['Astra', 'Corsa'],
    'Renault': ['Clio', 'Fluence'],
    'Toyota': ['Corolla'],
    'Volkswagen': ['Golf', 'Polo']
}

car_engines = {
    '80': [1.6, 1.8, 1.9],
    '100': [2.0],
    'A1': [1.4, 1.6],
    'A3': [1.8, 2.0],
    'A4': [2.0],
    'A5': [2.0],
    'A6': [2.0],
    'A8': [2.0],
    'TT': [2.0],
    'BX': [1.0],
    'C-Elysee': [1.2, 1.5, 1.6],
    'C1': [1.0, 1.4],
    'C2': [1.4, 1.6],
    'C3': [1.2, 1.4, 1.5, 1.6],
    'C4': [1.2, 1.4, 1.5, 1.6, 2.0],
    'C5': [1.6, 2.0],
    'Evasion': [1.9],
    'Saxo': [1.1, 1.4, 1.5, 1.6],
    'Xsara': [1.4, 1.6, 1.8, 1.9, 2.0],
    'ZX': [1.4, 1.8],
    'Linea': [1.3, 1.4, 1.6],
    'Egea': [1.0, 1.3, 1.5, 1.6],
    'Fiesta': [1.0, 1.25, 1.3, 1.4, 1.5, 1.6],
    'Focus': [1.4, 1.5, 1.6, 1.8, 2.0],
    'Astra': [1.0, 1.2, 1.3, 1.4, 1.5, 1.6],
    'Corsa': [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    'Clio': [0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8],
    'Fluence': [1.5, 1.6],
    'Corolla': [1.2, 1.3, 1.33, 1.4, 1.5, 1.6, 1.8],
    'Golf': [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0],
    'Polo': [1.0, 1.2, 1.4, 1.6, 1.9]
}

def get_models(brand):
    return brand_models.get(brand, [])

def get_engine(model):
    # Return the engine options for the specified model from the 'car_engines' dictionary
    return car_engines.get(model, [])

# Streamlit app navigation
navigation = st.sidebar.radio("Navigation", ["Home", "About Us", "Prediction (AI)"])

if navigation == 'Home':
    home_page()

elif navigation == "Prediction (AI)":
    # Add sliders, text inputs, or any other widgets for user input
    st.title('CPP Prediction Results')
    min_year, max_year = df_original['Vehicle Age'].min(), df_original['Vehicle Age'].max()
    brand = st.sidebar.selectbox('Select Car Brand', list(brand_map.keys()))
    available_models = get_models(brand)
    model = st.sidebar.selectbox('Select Car Model', available_models)
    selected_year = st.sidebar.selectbox('Select Car Year', options=list(range(min_year, max_year + 1)))
    car_age = current_year - selected_year
    color = st.sidebar.selectbox('Select Car Color', list(color_map.keys()))
    kilometers = st.sidebar.number_input('Select Car Kilometers', min_value=X['Kilometers'].min(),
                                         max_value=X['Kilometers'].max(), value=int(X['Kilometers'].mean()))
    available_engines = get_engine(model)  # Get available engines for the selected model

    engine = st.sidebar.selectbox('Select Engine Size', available_engines)

    # Define a predict button
    predict_button = st.sidebar.button('Predict')

    if predict_button:
        # Initialize and train the model
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_train, y_train)

        # Retrieve the numeric code for the selected color, brand, and model
        color_numeric = color_map[color]
        brand_numeric = brand_map[brand]
        model_numeric = model_map[model]

        brand_model = {key: value for key, value in model_map.items() if key.startswith("A")}
        # Perform prediction
        user_input = [[car_age, kilometers, engine, brand_numeric, model_numeric]]
        prediction = rf.predict(user_input)

        # Display prediction result
        # st.write('## Car Price Prediction')
        st.write(f' ### The predicted price for the selected car is: ')
        # st.write(f'## {prediction[0]:,.2f} TL'.replace(",", "X").replace(".", ",").replace("X", "."))
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h2 style="color: brown;">{prediction[0]:,.2f} TL</h2>
            </div>
            """.replace(",", "X").replace(".", ",").replace("X", "."),
            unsafe_allow_html=True
        )
        if brand == 'Audi' and model == '80':
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Audi80-1992.JPG/1200px-Audi80-1992.JPG', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=hD0SfDahnrQ')
        elif brand == 'Audi' and model == '100':
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJXBSpqJj_CdXQ2Dc2GEVDBseu1sILZTcRn7cmIeGLsg&s', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=4JYA4hTyygQ')
        elif brand == 'Audi' and model == 'A1':
            st.image('https://upload.wikimedia.org/wikipedia/commons/a/ad/2018_Audi_A1_S_Line_30_TFSi_S-A_1.0.jpg', use_column_width = True)
            st.video('https://www.youtube.com/watch?v=J4DvaEqCLw0')
        elif brand == "Audi" and model == 'A3':
            st.image('https://cdn.motor1.com/images/mgl/AebbV/s3/audi-a3-sportback-45-tfsi-e-2021.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=fueHz41q26o')
        elif brand == "Audi" and model == 'A4':
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR9WtJ8rFdKMB5ApyxuexI6rZ-so3ilnx4NEjTASFD-w&s', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=uDAIfOWAwUc')
        elif brand == "Audi" and model == 'A5':
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTonxycbySRF59CCDBb1dbHm0_X-0rzS-eeu6igwrGftQ&s', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=yPXBIgGHOn4')
        elif brand == "Audi" and model == 'A6':
            st.image('https://upload.wikimedia.org/wikipedia/commons/5/5f/2018_Audi_A6_TDi_Quattro_Front.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=4K4Is06NRfk')
        elif brand == "Audi" and model == 'A8':
            st.image('https://upload.wikimedia.org/wikipedia/commons/e/ea/Audi_A8_L_D5_IMG_0067.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=ya-DD1dEv38')
        elif brand == "Citroen" and model == 'BX':
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Citroen_BX_front_20080621.jpg/1200px-Citroen_BX_front_20080621.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=Nt6NWiC66bs')
        elif brand == "Citroen" and model == 'C-Elysee':
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Citro%C3%ABn_C-Elys%C3%A9e_-_prawy_prz%C3%B3d_%28MSP17%29.jpg/1280px-Citro%C3%ABn_C-Elys%C3%A9e_-_prawy_prz%C3%B3d_%28MSP17%29.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=l7FLPBnx_4M')
        elif brand == "Citroen" and model == 'C1':
            st.image('https://i0.shbdn.com/photos/74/43/66/x5_1058744366rs8.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=XEAs46OZ4VU')
        elif brand == "Citroen" and model == 'C2':
            st.image('https://cdn.motor1.com/images/mgl/6ZNQ0e/s3/citroen-c2-2003-2009.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=zWaRV6uL15A')
        elif brand == "Citroen" and model == 'C3':
            st.image('https://cdnuploads.aa.com.tr/uploads/sirkethaberleri/Contents/2023/04/07/thumbs_b_c_d759724c43a50ecc1f488cc949ec6585.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=131RrlN1BHA')
        elif brand == "Citroen" and model == 'C4':
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Citroen_C4_%282020%29_IMG_4202.jpg/1200px-Citroen_C4_%282020%29_IMG_4202.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=AQpODBsqnAM')
        elif brand == "Citroen" and model == 'C5':
            st.image('https://cdn.motor1.com/images/mgl/Q1On0/s3/2021-citroen-c5-x.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=d4ZjKECqZIU')
        elif brand == "Citroen" and model == 'Evasion':
            st.image('https://cdn3.focus.bg/autodata/i/citroen/evasion/evasion-u6u/large/b85cbcf5a0be676bef969951ed382cd3.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=mWq8wbx_AzI')
        elif brand == "Citroen" and model == 'Saxo':
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOy9I5hf79Q7rtpwAZN7xc3Ar--Ek3HBEP6uO2DDoYdw&s', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=cKS7o0dgsec')
        elif brand == "Citroen" and model == 'Xsara':
            st.image('https://i0.shbdn.com/photos/44/52/88/x5_1151445288zb3.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=CBVFFTG6cG4')
        elif brand == "Citroen" and model == 'ZX':
            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Citroen_ZX_front_20080118.jpg/1200px-Citroen_ZX_front_20080118.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=EMpYoyQplXk')
        elif brand == "Fiat" and model == 'Linea':
            st.image('https://cdn.alanyatekmar.com/0f72cc64-adba-4572-838b-fec4faca450c.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=rB86JUHpczA')
        elif brand == "Fiat" and model == 'Egea':
            st.image('https://www.gursesoto.com.tr/images/EGEA_SEDAN_LOUNGE.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=DujfIZ-0T64')
        elif brand == "Ford" and model == 'Focus':
            st.image('https://www.ford.com.tr/getmedia/19f61f3e-f21b-40c0-b973-419666594274/focus-renk-akik-siyah_1.jpg.aspx?width=1600&height=900&ext=.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=crPlZDKIIkI')
        elif brand == "Ford" and model == 'Fiesta':
            st.image('https://upload.wikimedia.org/wikipedia/commons/a/a7/Ford_Fiesta_ST-Line_%28VII%2C_Facelift%29_%E2%80%93_f_30012023.jpg',use_column_width=True)
            st.video('https://www.youtube.com/watch?v=H0tpvprPwI8')
        elif brand == "Opel" and model == 'Astra':
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZmquT5a260V1tQTc2NbkjaB4UT9g4TuGtP1IoJKqJig&s', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=1SY3wW3TfnI')
        elif brand == "Opel" and model == 'Corsa':
            st.image('https://stellantis3.dam-broadcast.com/medias/domain12808/media107258/2177697-ijug7m1bda-whr.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=QFzEVtY_1lQ')
        elif brand == "Renault" and model == 'Clio':
            st.image('https://stendustricomtr.teimg.com/crop/1280x720/stendustri-com-tr/images/haberler/2020/01/yeni_renault_clio_turkiye_pazarina_geliyor_h104344_5bba2.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=uiyjIIIYws4')
        elif brand == "Renault" and model == 'Fluence':
            st.image('https://cdn1.ntv.com.tr/gorsel/VCul8ampQ0azZ9ufb6TfLA.jpg?width=952&height=540&mode=both&scale=both', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=9lWpb8vNr4s')
        elif brand == "Toyota" and model == 'Corolla':
            st.image('https://scene7.toyota.eu/is/image/toyotaeurope/CORS0001b_2023-1?wid=1280&fit=fit,1&ts=1676906591441&resMode=sharp2&op_usm=1.75,0.3,2,0', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=UqSw9RcMAyw')
        elif brand == "Volkswagen" and model == 'Golf':
            st.image('https://cdn.motor1.com/images/mgl/3KVg1/s1/2022-volkswagen-golf-r-exterior.jpg', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=A2gnwh7qYCk')
        elif brand == "Volkswagen" and model == 'Polo':
            st.image('https://static.daktilo.com/sites/71/uploads/2021/09/23/2021-volkswagen-polo-fiyati-dudak-ucuklatiyor-gorenler-sasip-kaliyor-1.png', use_column_width=True)
            st.video('https://www.youtube.com/watch?v=WyawJQPxQ4M')
elif navigation == "About Us":
    about_us()