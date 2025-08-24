# 📊 Customer Lifetime Value (CLV) & RFM Segmentation App  

An **interactive Streamlit web application** for **Customer Segmentation (RFM Analysis)** and **Customer Lifetime Value (CLV) Prediction**.  
This project combines **machine learning (Random Forest, Gradient Boosting, XGBoost, LightGBM)** with **SHAP explainability** to help businesses understand, segment, and predict customer value.  

---

## 🚀 Features  
✅ **Upload Data**: Supports CSV/XLSX files with customer transaction data  
✅ **RFM Segmentation**: Calculates **Recency, Frequency, Monetary (RFM)** metrics and assigns customer segments  
✅ **CLV Prediction**: Predicts CLV using an **ensemble ML model** (RF, GBM, XGBoost, LightGBM)  
✅ **Visualizations**: Interactive plots for segment distribution and CLV analysis  
✅ **Explainability**: SHAP force plots & feature importance for **model transparency**  
✅ **Export Results**: Download full results as **CSV** and **PDF reports**  

---

## 📂 File Structure  

```bash
├── clv.py                     # Main Streamlit app (RFM + CLV Analysis)
├── CLTV.ipynb                 # Jupyter Notebook (EDA, model training, SHAP analysis)
├── data/                      # Example datasets
│   ├── online_retail_II 2009-2010csv.csv
├── outputs/                   # Generated outputs (reports, SHAP, etc.)
│   ├── *_force_plot.html
│   ├── *_dependence_*.png
│   ├── *_SHAP_Values.csv
│   ├── Final_Model_Report.pdf
````

---

## ⚙️ Installation

### 1️⃣ Requirements

* Python **3.8+**

### 2️⃣ Install Dependencies

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn joblib fpdf pillow shap xgboost lightgbm
```

---

## ▶️ Running the App

```bash
streamlit run clv.py
```

---

## 📝 How to Use

1. **Upload Data**

   * Supported formats: `CSV` / `XLSX`
   * Required columns:

     * `customerid`: Unique customer identifier
     * `invoiceno`: Invoice/transaction number
     * `invoicedate`: Date of transaction
     * `totalamount`: Transaction value

2. **View RFM Segmentation**

   * App calculates **Recency, Frequency, Monetary values**
   * Assigns customer segments automatically

3. **Predict CLV**

   * Predicts **Customer Lifetime Value (CLV)** for each customer
   * Uses a **pre-trained ensemble ML model**

4. **Visualize Results**

   * Explore:

     * Segment Distribution 📊
     * CLV Histograms 📈
     * Feature Importance 🔑
     * SHAP Force Plots ⚡

5. **Download Reports**

   * Export results as **CSV** (`RFM_CLV_Segmented_Customers.csv`)
   * Generate **PDF report** (`RFM_CLV_Report.pdf`)

---

## 📊 Data Preparation

Ensure your dataset includes these **mandatory columns**:

| Column        | Description                |
| ------------- | -------------------------- |
| `customerid`  | Unique customer identifier |
| `invoiceno`   | Invoice/transaction number |
| `invoicedate` | Date of transaction        |
| `totalamount` | Transaction value          |

👉 The app will **auto-clean & preprocess** the data for you.

---

## 🧠 Model Training & Explainability

* Detailed in `CLTV.ipynb`
* Models used:

  * 🌳 Random Forest
  * 📈 Gradient Boosting
  * ⚡ XGBoost
  * 💡 LightGBM
* **Explainability**:

  * SHAP Force Plots (`*_force_plot.html`)
  * Feature Dependence (`*_dependence_*.png`)
  * SHAP Values (`*_SHAP_Values.csv`)

---

## 📤 Outputs

* 📑 **CSV**: Full RFM & CLV results for all customers
* 📕 **PDF**: Downloadable summary report with tables & charts
* 📊 **Visualizations**: Segment distribution, CLV distribution, feature importance
* 🔍 **SHAP**: Force plots & dependence plots for model explainability

---

## 🧪 Example Workflow

1. Upload dataset: `online_retail_II 2009-2010csv.csv`
2. View RFM segments & predicted CLV
3. Download:

   * `RFM_CLV_Segmented_Customers.csv`
   * `RFM_CLV_Report.pdf`
4. Explore SHAP plots in:

   * `Random_Forest_force_plot.html`
   * `XGBoost_dependence_plot.png`

---

## 📚 References

* `clv.py` → Main app
* `CLTV.ipynb` → EDA, training & SHAP
* `Final_Model_Report.pdf` → Sample PDF output

---

## 📜 License

📌 This project is for **academic and non-commercial use only**.

---

## 📬 Contact

👤 **Soumya Ranjan Mohapatra**
📧 Email: [soumyasrm04@gmail.com](mailto:soumyasrm04@gmail.com)
🌐 LinkedIn: [srmohapatra](https://www.linkedin.com/in/srmohapatra)
💻 GitHub: [yourusername](https://github.com/yourusername)

---

✨ *"Segment. Predict. Explain. Empower businesses with data."* ✨

# 📊 Customer Lifetime Value (CLV) & RFM Segmentation App  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)  
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EB5E00)  
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-31D17B)  
![Power BI](https://img.shields.io/badge/Power%20BI-Visualization-F2C811?logo=powerbi&logoColor=black)  
![SHAP](https://img.shields.io/badge/Explainability-SHAP-EA1D2C)  

An **interactive Streamlit web application** for **Customer Segmentation (RFM Analysis)** and **Customer Lifetime Value (CLV) Prediction**.  
This project combines **machine learning (Random Forest, Gradient Boosting, XGBoost, LightGBM)** with **SHAP explainability** to help businesses understand, segment, and predict customer value.  

