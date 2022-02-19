import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from apps import praproses
import os
from dateutil import parser
def app():
  if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Home` page!")
  else:
    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameter'):
      split_size = st.sidebar.slider('Rasio Pembagian Data (% Untuk Data Latih)', 10, 90, 80, 5)
      jumlah_fitur = st.sidebar.slider('jumlah pilihan fitur (Untuk Data Latih)', 5, 47, 20, 5)
      parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 10, 100, 50, 10)
      k = st.sidebar.slider('Jumlah K (KNN)', 2, 10, 5, 1)
      
    if st.sidebar.button('Press to Train'):  
      df = pd.read_csv('data/main_data.csv')
      df1=pd.read_csv('data/df1.csv')
      from sklearn.feature_selection import mutual_info_classif
      #determine the mutual information
      mutual_info = mutual_info_classif(df.drop(columns=['enrolled']), df.enrolled)
      mutual_info = pd.Series(mutual_info)
      mutual_info.index = df.drop(columns=['enrolled']).columns
      mutual_info.sort_values(ascending=False)
      from sklearn.feature_selection import SelectKBest
      fitur_terpilih = SelectKBest(mutual_info_classif, k=jumlah_fitur)
      fitur_terpilih.fit(df.drop(columns=['enrolled']), df.enrolled)
      pilhan_kolom = df.drop(columns=['enrolled']).columns[fitur_terpilih.get_support()]
      pd.Series(pilhan_kolom).to_csv('data/fitur_pilihan.csv',index=False)
      fitur = pilhan_kolom.tolist()
      baru = df[fitur]
      from sklearn.preprocessing import StandardScaler
      sc_X = StandardScaler()
      pilhan_kolom = sc_X.fit_transform(baru)
      import joblib
      joblib.dump(sc_X, 'data/minmax_scaler.joblib')
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(pilhan_kolom, df['enrolled'],test_size=(100-split_size)/100, random_state=111)
      st.write(X_test)
      from sklearn.metrics import accuracy_score
      from sklearn.metrics import matthews_corrcoef
      from sklearn.metrics import f1_score
      from sklearn.naive_bayes import GaussianNB
      nb = GaussianNB() # Define classifier)
      nb.fit(X_train, y_train)

      # Make predictions
      y_train_pred = nb.predict(X_train)
      y_test_pred = nb.predict(X_test)

      # Training set performance
      nb_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
      nb_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
      nb_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

      # Test set performance
      nb_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
      nb_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
      nb_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

      st.write('Model performance for Training set')
      st.write('- Accuracy: %s' % nb_train_accuracy)
      st.write('- MCC: %s' % nb_train_mcc)
      st.write('- F1 score: %s' % nb_train_f1)
      st.write('----------------------------------')
      st.write('Model performance for Test set')
      st.write('- Accuracy: %s' % nb_test_accuracy)
      st.write('- MCC: %s' % nb_test_mcc)
      st.write('- F1 score: %s' % nb_test_f1)
      st.write('++++++++++++++++++++++++++++++++++')
      from sklearn.ensemble import RandomForestClassifier

      rf = RandomForestClassifier(n_estimators=parameter_n_estimators) # Define classifier
      rf.fit(X_train, y_train) # Train model

      # Make predictions
      y_train_pred = rf.predict(X_train)
      y_test_pred = rf.predict(X_test)

      # Training set performance
      rf_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
      rf_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
      rf_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

      # Test set performance
      rf_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
      rf_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
      rf_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

      st.write('Model performance for Training set')
      st.write('- Accuracy: %s' % rf_train_accuracy)
      st.write('- MCC: %s' % rf_train_mcc)
      st.write('- F1 score: %s' % rf_train_f1)
      st.write('----------------------------------')
      st.write('Model performance for Test set')
      st.write('- Accuracy: %s' % rf_test_accuracy)
      st.write('- MCC: %s' % rf_test_mcc)
      st.write('- F1 score: %s' % rf_test_f1)
      st.write('++++++++++++++++++++++++++++++++++')
      from sklearn.ensemble import StackingClassifier
      from sklearn.neighbors import KNeighborsClassifier

      estimator_list = [
          ('nb',nb),
          ('rf',rf)]

      # Build stack model
      stack_model = StackingClassifier(
          estimators=estimator_list, final_estimator=KNeighborsClassifier(k),cv=5
      )

      # Train stacked model
      stack_model.fit(X_train, y_train)

      # Make predictions
      y_train_pred = stack_model.predict(X_train)
      y_test_pred = stack_model.predict(X_test)

      # Training set model performance
      stack_model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
      stack_model_train_mcc = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
      stack_model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score

      # Test set model performance
      stack_model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
      stack_model_test_mcc = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
      stack_model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

      st.write('Model performance for Training set')
      st.write('- Accuracy: %s' % stack_model_train_accuracy)
      st.write('- MCC: %s' % stack_model_train_mcc)
      st.write('- F1 score: %s' % stack_model_train_f1)
      st.write('----------------------------------')
      st.write('Model performance for Test set')
      st.write('- Accuracy: %s' % stack_model_test_accuracy)
      st.write('- MCC: %s' % stack_model_test_mcc)
      st.write('- F1 score: %s' % stack_model_test_f1)
      st.write('++++++++++++++++++++++++++++++++++')
      acc_train_list = {'nb':nb_train_accuracy,
      'rf': rf_train_accuracy,
      'stack': stack_model_train_accuracy}

      mcc_train_list = {'nb':nb_train_mcc,
      'rf': rf_train_mcc,
      'stack': stack_model_train_mcc}

      f1_train_list = {'nb':nb_train_f1,
      'rf': rf_train_f1,
      'stack': stack_model_train_f1}

      acc_df = pd.DataFrame.from_dict(acc_train_list, orient='index', columns=['Accuracy'])
      mcc_df = pd.DataFrame.from_dict(mcc_train_list, orient='index', columns=['MCC'])
      f1_df = pd.DataFrame.from_dict(f1_train_list, orient='index', columns=['F1'])
      df = pd.concat([acc_df, mcc_df, f1_df], axis=1)
      st.write(df)
      
      import joblib
      joblib.dump(stack_model, 'data/stack_model.pkl')

      var_enrolled = df1['enrolled']
      #membagi menjadi train dan test untuk mencari user id
      X_train, X_test, y_train, y_test = train_test_split(df1, df1['enrolled'], test_size=(100-split_size)/100, random_state=111)
      train_id = X_train['user']
      test_id = X_test['user']
      #menggabungkan semua
      y_pred_series = pd.Series(y_test).rename('asli',inplace=True)
      hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
      hasil_akhir['prediksi']=y_test_pred
      hasil_akhir = hasil_akhir[['user','asli','prediksi']].reset_index(drop=True)
      st.write(hasil_akhir)
