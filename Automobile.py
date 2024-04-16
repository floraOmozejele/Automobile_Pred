import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('automobile.csv') 

st.markdown("<h1 style = 'color: #EFBC9B; text-align: center; font-size: 60px; font-family:Helvetica'>AUTOMOBILE PRICE PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #5BBCFF; text-align: center; font-family: cursive '>Build By Salmon Crushers</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

#Add an Image 
st.image('pngwing.com (14).png', caption = 'Built by Salmon')

# Add Project Problem body-stylement 
st.markdown("<h2 style = 'color: #5BBCFF; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("<p>By meticulously analyzing key parameters such as Manufacturing Expense, Administrative Expense, and Research and Development Spending, our team is committed to crafting a sophisticated predictive model tailored for the automotive sector. This endeavor seeks to furnish investors, stakeholders, and automotive industry leaders with actionable, data-driven insights. Additionally, this initiative serves as an invaluable resource for aspiring entrepreneurs within the automotive industry, offering a comprehensive framework to assess the feasibility of their business models and strategically enhance their operations for sustained long-term succe</p>", unsafe_allow_html= True)

#SideBar Design
st.sidebar.image('pngwing.com (15).png')

st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.divider()
st.header("Project Data")
st.dataframe(data, use_container_width = True)

#User Inputs
# user inputs
curb_weight = st.sidebar.number_input("curb-weight", placeholder='insert your numbers...')
normalized_losses = st.sidebar.number_input ('normalized-losses', data ['normalized-losses'].min(), data['normalized-losses'].max())
make = st.sidebar.selectbox ('make', data['make'].unique())
horsepower = st.sidebar.number_input ("horsepower", placeholder='insert your numbers...')
city = st.sidebar.number_input('city-mpg', placeholder = 'insert your city...')
height = st.sidebar.number_input("height", placeholder='insert your numbers...')
body_style = st.sidebar.selectbox ('body-style', data['body-style'].unique())
#curb = st.sidebar.number_input('fuel_system', data['fuel_system'].min(), data['fuel_system'].max())
#norm = st.sidebar.number_input('fueltype', data['fueltype'].min(), data['fueltype'].max())
#make = st.sidebar.selectbox('make', data['make'].unique())
#body= st.sidebar.selectbox('body-style', data['body-style'].unique())
#horse = st.sidebar.selectbox('horsepower', data['horsepower'].unique())
#city = st.sidebar.number_input('city-mpg', data['city-mpg'].min(), data['city-mpg'].max())
#height = st.sidebar.number_input('height', data['height'].min(), data['height'].max())



#users input
input_var = pd.DataFrame()
input_var['curb-weight'] = [curb_weight]
input_var['normalized-losses'] = [normalized_losses]
input_var['make'] = [make]
input_var['body-style'] = [body_style]
input_var['horsepower'] = [horsepower]
input_var['city-mpg'] = [city]
input_var['height'] = [height]


st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width=True)


# Import Transformers
make = joblib.load('make_encoder.pkl')
body = joblib.load('body-style_encoder.pkl')
#horse = joblib.load('horsepower_encoder.pkl')

# Transform users input according to training scale and encoding 
# transform users input according to training scale and encoding
# transform the users input with the imported scalers
input_var['make'] = make.transform(input_var[['make']])
input_var['body-style'] = body.transform(input_var[['body-style']])
#input_var['horsepower'] = horse.transform(input_var[['horsepower']])




# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)
    # Modeling 
model = joblib.load('AutomobilModel.pkl')

#to have a button for the user
if st.button('Predict Price'):
    predicted_price = model.predict(input_var)
    st.success(f"The Price of this Car is  {predicted_price[0].round()}")

    # Modeling 
#model = joblib.load('AutomobilModel.pkl')