from PIL import Image  # Importing Image with an alias to avoid conflicts
import streamlit as st


def home_page():
    st.title("")
    st.write("")
    col1, col2 = st.columns([1, 5])
    with col1:
        # Load and display CPP logo
        cpp_logo = 'CN.png'
        img = Image.open(cpp_logo)
        st.image(img, width=100)
    with col2:
        # Add "Car Price Predictor" text
        st.title("Car Price Predictor")
    st.write('''
        "Welcome to our Car Price Predictor app! This platform utilizes advanced machine learning techniques to provide 
        accurate predictions of car prices based on various factors such as year, kilometers driven, engine size, color,
         brand, and model. Simply input your desired car specifications, and our model will generate a prediction for you. 
         Whether you're buying, selling, or simply curious about car prices, our app is here to assist you in making 
         informed decisions. Get started now and discover the predicted price for your dream car!"
    ''')
    car_images = [
        'https://hips.hearstapps.com/hmg-prod/images/2024-audi-rs7-performance-motion-front-2-1669663936.jpg?crop=0.689xw:0.517xh;0.276xw,0.368xh&resize=1200:*',
        'https://scene7.toyota.eu/is/image/toyotaeurope/Yeni-Toyota-Corolla-1:Medium-Landscape?ts=0&resMode=sharp2&op_usm=1.75,0.3,2,0',
        'https://arabam-blog.mncdn.com/wp-content/uploads/2024/02/Volkswagen-1.jpg',
        'https://image5.sahibinden.com/staticContent/vehicleStockImagesV2/1240/1769cb/x_1216_684_68391001240-4538416c2ded23bb7ce9dabef872b26d.jpg'
    ]
    for img_path in car_images:
        st.image(img_path, use_column_width=True)