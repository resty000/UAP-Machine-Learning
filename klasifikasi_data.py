import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
import xgboost as xgb  # type: ignore
import torch
import joblib
import os
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

# Function to create Feedforward Neural Network (FNN) for multi-class classification
def create_fnn(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for multi-class
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to initialize and train models with feature importance
def train_model(algorithm, X_train, y_train, X_test, y_test, model_folder):
    model_file = None
    predictions = None
    feature_importance = None  # For storing feature importance

    if algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model_file = os.path.join(model_folder, "model_logistic_regression.joblib")

    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
        model_file = os.path.join(model_folder, "model_decision_tree.joblib")

    elif algorithm == "Random Forest":
        model = RandomForestClassifier()
        model_file = os.path.join(model_folder, "model_random_forest.joblib")

    elif algorithm == "SVM":
        model = LinearSVC()  # Faster alternative to SVC
        model_file = os.path.join(model_folder, "model_svm.joblib")

    elif algorithm == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model_file = os.path.join(model_folder, "model_neural_network.joblib")

    elif algorithm == "TabNet":
        model = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            verbose=0
        )
        model_file = os.path.join(model_folder, "model_tabnet.joblib")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    elif algorithm == "Feedforward Neural Network (FNN)":
        num_classes = len(np.unique(y_train))
        y_train_fnn = to_categorical(y_train, num_classes=num_classes)
        y_test_fnn = to_categorical(y_test, num_classes=num_classes)

        model = create_fnn(X_train.shape[1], num_classes)
        model.fit(X_train, y_train_fnn, validation_data=(X_test, y_test_fnn), epochs=20, batch_size=32, verbose=1)
        predictions = np.argmax(model.predict(X_test), axis=-1)  # Convert back to labels

        model_file = os.path.join(model_folder, "model_fnn.joblib")

    elif algorithm == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        model_file = os.path.join(model_folder, "model_xgboost.joblib")

    # Train and evaluate model (excluding FNN and TabNet)
    if algorithm not in ["Feedforward Neural Network (FNN)", "TabNet"]:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # If the model has feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):  # For Logistic Regression
            feature_importance = model.coef_[0]

        # Save model to file
        joblib.dump(model, model_file)

    return model_file, predictions, feature_importance

# Function to display feature importance in the app as a bar plot
def display_feature_importance(feature_importance, feature_columns):
    if feature_importance is not None:
        # Create a DataFrame for Feature Importance
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importance
        })

        # Sort the features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Display the feature importance as a bar plot
        st.subheader("Feature Importance (Grafik):")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)

# Classification App
def app():
    st.header("Klasifikasi")

    # Inisialisasi session state untuk data
    if 'data' not in st.session_state:
        st.session_state.data = None

    # Pastikan data tersedia sebelum melanjutkan
    if st.session_state.data is not None:
        # Multiselect untuk memilih kolom fitur
        feature_columns = st.multiselect(
            "Pilih kolom fitur:",
            options=st.session_state.data.columns,  # Semua kolom dapat dipilih
        )

        # Selectbox untuk memilih kolom target
        target_column = st.selectbox(
            "Pilih kolom target:",
            options=st.session_state.data.columns,  # Semua kolom dapat dipilih
            help="Pilih kolom target yang ingin diprediksi"
        )

        # Validasi input
        if not feature_columns:
            st.warning("Harap pilih setidaknya satu kolom sebagai fitur.")
            return

        if target_column in feature_columns:
            st.warning("Kolom target tidak boleh sama dengan kolom fitur.")
            return

        # Pilih algoritma
        algorithm = st.selectbox(
            "Pilih algoritma:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Neural Network", "TabNet", "Feedforward Neural Network (FNN)", "XGBoost"]
        )

        # Tombol untuk memulai klasifikasi
        if st.button("Mulai Klasifikasi"):
            try:
                # Memulai proses klasifikasi
                X = st.session_state.data[feature_columns]
                y = st.session_state.data[target_column]

                # Handling missing values with fillna
                X.fillna("Unknown", inplace=True)

                # Konversi kolom kategori ke numerik
                label_encoders = {}
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le

                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)

                # Normalisasi data
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Pastikan folder untuk menyimpan model tersedia
                model_folder = "C:/Users/Resty Putri SuciYani/Downloads/UAP_ML/src/modell"
                os.makedirs(model_folder, exist_ok=True)

                # Train and evaluate model
                model_file, predictions, feature_importance = train_model(algorithm, X_train, y_train, X_test, y_test, model_folder)

                # Tampilkan akurasi
                accuracy = accuracy_score(y_test, predictions)
                st.success(f"Akurasi Model ({algorithm}): {accuracy:.2f}")

                # Tampilkan laporan klasifikasi sebagai tabel
                report_dict = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.subheader("Laporan Klasifikasi:")
                st.table(report_df)

                # Display feature importance if available
                display_feature_importance(feature_importance, feature_columns)

                st.success(f"Model disimpan di: {model_file}")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data.")
