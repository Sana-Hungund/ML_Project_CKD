
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('D://Machine/vvv.csv')


st.title('Prediction of Chronic Kidney Disease (CKD)')
st.markdown('<style>h1{color: #AF601A;}</style>', unsafe_allow_html=True)
st.write('')
image = Image.open('C://Users/Asus/Pictures/Screenshots/kidney_img.png')

st.image(image)

st.write('')
st.sidebar.header('Input Patient Data')




x = df.drop(['classification'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



def user_report():
  age=st.sidebar.number_input('age',min_value=2, max_value=100, value=22, step=1, format=None, key=None)
  bp=st.sidebar.number_input('blood pressure (mm/Hg)', min_value=45, max_value=180, value=66, step=1)
  sg=st.sidebar.number_input('urin specific gravity (sg)',min_value=1.005,max_value=1.025,step=0.005)
  al=st.sidebar.selectbox('albumin (al): yes (1) no (0)',options=[0,1,2,3,4,5])
  su=st.sidebar.selectbox('sugar (su): yes (1) no (0)',options=[0,1,2,3,4,5])
  rbc=st.sidebar.selectbox('red blood care (rbc): abnormal (1) normal (0)',options=[0,1])
  pc=st.sidebar.selectbox('pus cell (pc): abnormal (1) normal (0)',options=[0,1])
  pcc=st.sidebar.selectbox('pus cell clumps (pcc): present (1) not present (0)',options=[0,1])
  ba=st.sidebar.selectbox('bacteria (ba): present (1) not present (0)',options=[0,1])
  bgr=st.sidebar.number_input('blood glucose random (mgs/dl)',min_value=70, max_value=500, value=131, step=1)
  bu=st.sidebar.number_input('blood urea (mgs/dl)',min_value=10, max_value=309, value=52, step=1)
  sc=st.sidebar.number_input('serum creatinine (mgs/dl)',min_value=0.4, max_value=15.2, value=2.2, step=0.1)
  sod=st.sidebar.number_input('sodium (mEq/L)',min_value=111, max_value=150, value=138, step=1)
  pot=st.sidebar.number_input('potassium (mEq/L)',min_value=2.5, max_value=47.0, value=4.6, step=0.1)
  hemo=st.sidebar.number_input('hemoglobin (gms)',min_value=3.1, max_value=17.8, value=13.7, step=0.1)
  pcv=st.sidebar.number_input('packed cell count (pcv)',min_value=16, max_value=55, value=30, step=1)
  wc=st.sidebar.number_input('white blood cell count (cells/cumm)',min_value=3000, max_value=15000, value=7000, step=100)
  rc=st.sidebar.number_input('red blood cell count (millions/cumm)',min_value=2.2, max_value=6.9, value=5.0, step=0.1)
  htn=st.sidebar.selectbox('hypertension (htn): yes (1) no (0)',options=[0,1])
  dm=st.sidebar.selectbox('diabetes mellitus (dm): yes (1) no (0)',options=[0,1])
  cad=st.sidebar.selectbox('coronary artery disease (cad): yes (1) no (0)',options=[0,1])
  appet=st.sidebar.selectbox('appetite (appet): good (1) poor (0)',options=[0,1])
  pe=st.sidebar.selectbox('pedal edema (pe): yes (1) no (0)',options=[0,1])
  ane=st.sidebar.selectbox('anemia (ane): yes (1) no (0)',options=[0,1])



  user_report_data = {
      'age':age,
      'bp':bp,
      'sg':sg,
      'al':al,
      'su':su,
      'rbc':rbc,
      'pc':pc,
      'pcc':pcc,
      'ba':ba,
      'bgr':bgr,
      'bu':bu,
      'sc':sc,
      'sod':sod,
      'pot':pot,
      'hemo':hemo,
      'pcv':pcv,
      'wc':wc,
      'rc':rc,
      'htn':htn,
      'dm':dm,
      'cad':cad,
      'appet':appet,
      'pe':pe,
      'ane':ane
  }


    
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
html_temp1 = """
<div style><h3 style="color:#333539 ;">Patient Data</h3>
</div>"""
st.markdown(html_temp1,unsafe_allow_html=True)

st.write(user_data)


rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.write('')

check_button = st.button('Check to see if you are at risk of KD?')
output=''
if check_button:
  html_temp = """
  <div style><h4 style="color:#214191 ;">According to the given parameters, the model predicts</h4>
  </div>"""
  st.markdown(html_temp,unsafe_allow_html=True)
  if user_result[0]==0:
    output = 'You are not at risk of CKD.'
    st.title(output)
  else:
    output = 'You might be at risk of CKD. Check with your doctor.'
    st.markdown('<style>h3{color: red;}</style>', unsafe_allow_html=True)
    st.subheader(output)
   

  st.title('Visualised Patient Report')
  if user_result[0]==0:
    color = 'blue'
  else:
    color = 'red'


# Age vs Bp
  original_title = '<p style=" color:#FFA109; font-size: 28px;">Blood Pressure Value Graph (Others vs Yours)</p>'
  st.markdown(original_title, unsafe_allow_html=True)
  fig_bp = plt.figure(figsize=(5, 3))
  ax5 = sns.scatterplot(x = 'age', y = 'bp', data = df)
  ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 80, color = color)
  plt.xticks(np.arange(10,100,10))
  plt.yticks(np.arange(0,130,20))
  st.pyplot(fig_bp)


  original_title1 = '<p style=" color:#FFA109; font-size: 28px;">Albumin Value Graph (Others vs Yours)</p>'
  st.markdown(original_title1, unsafe_allow_html=True)
  fig_bp = plt.figure(figsize=(5, 3))
  ax5 = sns.scatterplot(x = 'age', y = 'al', data = df)
  ax6 = sns.scatterplot(x = user_data['age'], y = user_data['al'], s = 80, color = color)
  plt.xticks(np.arange(0,100,10))
  plt.yticks(np.arange(0,6,1))
  st.pyplot(fig_bp)

  original_title2 = '<p style=" color:#FFA109; font-size: 28px;">Blood Glucose Random Value Graph (Others vs Yours)</p>'
  st.markdown(original_title2, unsafe_allow_html=True)
  fig_bp = plt.figure(figsize=(5, 3))
  ax5 = sns.scatterplot(x = 'age', y = 'bgr', data = df)
  ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bgr'], s = 80, color = color)
  plt.xticks(np.arange(10,100,10))
  plt.yticks(np.arange(50,500,50))
  st.pyplot(fig_bp)


