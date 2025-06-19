
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Menggunakan algoritma **Decision Tree** berdasarkan data IPK, Kehadiran, dan SKS")

# Load dataset
df = pd.read_csv("dataset_prediksi_kelulusan.csv")
st.subheader("📋 Dataset")
st.dataframe(df)

# Preprocessing
X = df[["IPK", "Kehadiran", "SKS"]]
y = df["Lulus"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Evaluasi Model")
st.write(f"**Akurasi:** {accuracy * 100:.2f}%")

# Classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Ya", "Tidak"])
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Ya", "Tidak"], yticklabels=["Ya", "Tidak"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
st.pyplot(fig_cm)

# Input Manual
st.subheader("📝 Input Data Mahasiswa")
ipk = st.slider("IPK", 2.0, 4.0, 3.0)
kehadiran = st.slider("Kehadiran (%)", 50, 100, 85)
sks = st.slider("Total SKS", 100, 160, 144)

input_data = pd.DataFrame([[ipk, kehadiran, sks]], columns=["IPK", "Kehadiran", "SKS"])

# Predict
if st.button("🔮 Prediksi Kelulusan"):
    hasil = model.predict(input_data)[0]
    st.success(f"Prediksi: Mahasiswa **{'lulus' if hasil == 'Ya' else 'tidak lulus'}** tepat waktu.")

# Decision Tree Visualization
st.subheader("🌳 Visualisasi Decision Tree")
fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
st.pyplot(fig_tree)
