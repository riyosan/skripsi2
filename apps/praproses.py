import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

def upload_dataset():
  # data=st.sidebar.file_uploader("Unggah File CSV", type=['CSV']).header('1. Unggah File CSV')
  with st.sidebar.header('1. Unggah File CSV'):
    data=st.sidebar.file_uploader("Unggah File CSV",type=['csv'])
  return data

def prepro(df1):
  #menghitung ulang isi dari kolom screen_litst karena tidak pas di kolom num screens
  df1['screen_list'] = df1.screen_list.astype(str) + ','
  df1['num_screens'] = df1.screen_list.str.count(',')
  #menghapus kolom numsreens yng lama
  df1.drop(columns=['numscreens'], inplace=True)
  container = st.columns((1.9, 1.1))
  df1_types = df1.dtypes.astype(str)
  with container[0]:
    st.write(df1)
    st.text('Merevisi kolom numscreens')
  with container[1]:
    st.write(df1_types)
    st.text('Tipe data setiap kolom')
  #feature engineering
  #karena kolom hour ada spasinya, maka kita ambil huruf ke 1 sampai ke 3
  df1.hour=df1.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open dan enrolled_date itu adalah string, maka perlu diubah ke datetime
  df1.first_open=[parser.parse(i) for i in df1.first_open]
  #didalam dataset orang yg belum langganan itu NAN, maka jika i=string biarin, klo ga string diuah ke datetime kolom nan nya biarin tetap nat
  df1.enrolled_date=[parser.parse(i) if isinstance(i, str)else i for i in df1.enrolled_date]
  #membuat kolom selisih , yaitu menghitung berapa lama orang yg firs_open menjadi enrolled
  df1['selisih']=(df1.enrolled_date-df1.first_open).astype('timedelta64[h]')
  #karna digrafik menunjukkan orang kebanyakan enroll selama 24 jam pertama, maka kalau lebih dari 24 jam dianggap ga penting
  df1.loc[df1.selisih>24, 'enrolled'] = 0
  container2 = st.columns((1.9, 1.1))
  df1_types = df1.dtypes.astype(str)
  with container2[0]:
    st.write(df1)
    st.text('Merevisi kolom hour')
  with container2[1]:
    st.write(df1_types)
    st.text('Tipe data setiap kolom')
  # mengambil dataset top screen
  top_screens=pd.read_csv('top_screens.csv')
  # diubah ke numppy arry dan mengambil kolom ke2 saja karna kolom1 isinya nomor
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  df = df1.copy()
  #mengubah isi dari file top screen menjadi numerik
  for i in top_screens:
    df[i]=df.screen_list.str.contains(i).astype(int)
  #semua item yang ada di file top screen dihilangkan dari kolom screen list
  for i in top_screens:
    df['screen_list']=df.screen_list.str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df['lainnya']=df.screen_list.str.count(',')
  st.write(df)
  st.text('Mengubah isi screen_list menjadi kolom baru')
  #menggabungkan item yang mirip mirip, seperti kredit 1 kredit 2 dan kredit 3
  #funneling = menggabungkan beberapa screen yang sama dan menghapus layar yang sama
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df['jumlah_loan']=df[layar_loan].sum(axis=1)
  df.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df['jumlah_loan']=df[layar_saving].sum(axis=1)
  df.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df['jumlah_credit']=df[layar_credit].sum(axis=1)
  df.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df['jumlah_cc']=df[layar_cc].sum(axis=1)
  df.drop(columns=layar_cc, inplace=True)
  #menghilangkan kolom yang ga relevan
  df_numerik=df.drop(columns=['user','first_open','screen_list','enrolled_date','selisih'], inplace=False)
  st.write(df_numerik)
  #membuat plot korelasi tiap kolom dengan enrolled
  korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
  plot=korelasi.plot.bar(title='korelasi variabel')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
  st.text('Membuat plot korelasi tiap koklom terhadap kelasnya(enrolled)')
  from sklearn.feature_selection import mutual_info_classif
  #determine the mutual information
  mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
  mutual_info.sort_values(ascending=False)
  mutual_info.sort_values(ascending=False).plot.bar(title='urutannya')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()
  st.text('mengurutkan korelasi setiap kolom terhadap kelasnya(enrolled)')
  df_numerik.to_csv('data/main_data.csv', index=False)
  df1.to_csv('data/df1.csv', index=False)
def app():
  global data
  filenya=upload_dataset()
  if filenya is not None:
    st.write("""### Fintech_dataset""")
    df1=pd.read_csv(filenya)
    st.write(df1)
    if st.button("Praproses Data"):
      prepro(df1)
  else:
    if st.sidebar.button('Klik Untuk Gunakan Data Sampel'):
      df1 = pd.read_csv('fintech_data.csv')
      prepro(df1)
