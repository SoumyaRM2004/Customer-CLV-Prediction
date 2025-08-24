import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import os
import re
import tempfile
from io import BytesIO
from fpdf import FPDF
from PIL import Image
import io
import time
import shutil
from contextlib import contextmanager
import uuid
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


st.set_page_config(page_title="🧮 RFM & CLV Analyzer", layout="wide")
st.title("🧮 Customer Segmentation (RFM) & CLV Prediction App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv", "xlsx"])

# Load model
model_path = 'voting_ensemble_model2.pkl'
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()
model = joblib.load(model_path)

def save_fig_to_buffer(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def add_download_button(fig, label):
    buf = save_fig_to_buffer(fig)
    st.download_button(
        label=label,
        data=buf,
        file_name=f"{label.replace(' ','_')}.png",
        mime="image/png",
        key=label
    ) 

def generate_pdf_report(rfm_df, combined_df, fig1_buf, fig2_buf):
    print("generate_pdf_report called with args:")
    print("rfm_df:", type(rfm_df))
    print("combined_df:", type(combined_df))
    print("fig1_buf:", type(fig1_buf))
    print("fig2_buf:", type(fig2_buf))
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Customer Segmentation (RFM) & CLV Report", 0, 1, 'C')

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "RFM Segments Summary", 0, 1)

        # Add RFM table (top 10 rows)
        pdf.set_font("Arial", size=10)
        rfm_head = rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(10)
        col_widths = [30, 20, 20, 25, 40]

        # Table header
        headers = rfm_head.columns.tolist()
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, 1, 0, 'C')
        pdf.ln()

        # Table rows
        for _, row in rfm_head.iterrows():
            pdf.cell(col_widths[0], 6, str(row['CustomerID']), 1)
            pdf.cell(col_widths[1], 6, str(row['Recency']), 1, 0, 'C')
            pdf.cell(col_widths[2], 6, str(row['Frequency']), 1, 0, 'C')
            pdf.cell(col_widths[3], 6, f"${row['Monetary']:.2f}", 1, 0, 'R')
            pdf.cell(col_widths[4], 6, str(row['Segment']), 1, 0, 'C')
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "CLV Prediction Summary", 0, 1)

        clv_head = combined_df[['CustomerID', 'Segment', 'Predicted_CLV']].head(10)
        col_widths = [30, 40, 30]

        # Table header
        headers = clv_head.columns.tolist()
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, 1, 0, 'C')
        pdf.ln()

        # Table rows
        for _, row in clv_head.iterrows():
            pdf.cell(col_widths[0], 6, str(row['CustomerID']), 1)
            pdf.cell(col_widths[1], 6, str(row['Segment']), 1)
            pdf.cell(col_widths[2], 6, f"${row['Predicted_CLV']:.2f}", 1, 0, 'R')
            pdf.ln()

        # Add RFM Segment distribution plot
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Charts:", 0, 1)

        pdf.cell(0, 10, "Customer Segments Distribution", 0, 1)
        pdf.image(fig1_buf, x=10, w=190)
        pdf.ln(10)

        pdf.cell(0, 10, "CLV Analysis", 0, 1)
        pdf.image(fig2_buf, x=10, w=190)

        # Output pdf in memory buffer
        # Output PDF as bytes and store in buffer
        pdf_output = pdf.output(dest='S').encode('latin1')  # 'S' returns the content as a string
        pdf_buffer = BytesIO(pdf_output)
        return pdf_buffer

   
    except Exception as e:
            st.warning(f"Could not clean up temporary file: {e}")

# ✅ MAIN APP LOGIC STARTS HERE
if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📂 Original columns in your CSV:", df.columns.tolist())

        df.columns = [re.sub(r'\W+', '', col).strip().lower() for col in df.columns]

        rename_map = {
            'invoice': 'invoiceno',
            'customerno': 'customerid',
            'customerid': 'customerid',
            'customer_id': 'customerid',
            'custid': 'customerid',
            'invno': 'invoiceno',
            'invoiceno': 'invoiceno',
            'invoice_number': 'invoiceno',
            'billno': 'invoiceno',
            'billnumber': 'invoiceno',
            'billdate': 'invoicedate',
            'transdate': 'invoicedate',
            'date': 'invoicedate',
            'invoicedate': 'invoicedate',
            'amount': 'totalamount',
            'netamount': 'totalamount',
            'totalprice': 'totalamount',
            'totalamount': 'totalamount',
            'price': 'price',
            'unitprice': 'price',
            'qty': 'quantity',
            'quantity': 'quantity',
            'quantityordered': 'quantity'
        }

        df.rename(columns=lambda x: rename_map.get(x, x), inplace=True)
        st.write("🔄 Columns after normalization & renaming:", df.columns.tolist())

        if 'totalamount' not in df.columns:
            if ('quantity' in df.columns) and ('price' in df.columns):
                df['totalamount'] = df['quantity'] * df['price']
                st.info("ℹ️ 'TotalAmount' calculated as Quantity × Price.")
            else:
                st.error("❌ Missing 'totalamount' and cannot compute it due to missing 'quantity' or 'price'.")
                st.stop()


        required_cols = ['customerid', 'invoiceno', 'invoicedate', 'totalamount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Required columns missing: {missing_cols}")
            st.stop()

        # Convert invoicedate to datetime
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df.dropna(subset=['invoicedate'], inplace=True)

        df = df[required_cols]
        
        st.subheader("📋 Cleaned Sample Data")
        st.dataframe(df.head())

        # Compute RFM
        today = datetime.now()
        
        rfm = df.groupby('customerid').agg({
            'invoicedate': lambda x: (today - x.max()).days,
            'invoiceno': 'nunique',
            'totalamount': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
        rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
        rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
        rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

        def segment(row):
            if row['RFM_Score'] == '444':
                return 'Loyal'
            elif row['R'] == 4:
                return 'Recent'
            elif row['F'] == 4:
                return 'Frequent'
            elif row['M'] == 4:
                return 'Big Spender'
            elif row['R'] == 1:
                return 'At Risk'
            else:
                return 'Others'

        rfm['Segment'] = rfm.apply(segment, axis=1)

        st.subheader("🔍 RFM Segments")
        st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].sort_values(by='Recency'))

        # Plot segment distribution
        st.subheader("📊 RFM Segment Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='Set2', ax=ax1)
        ax1.set_title("Customer Segments")
        st.pyplot(fig1)
        add_download_button(fig1, "Download RFM Segment Distribution")

        # CLV Prediction
        st.subheader("📈 Predicting CLV...")

        clv_df = df.groupby('customerid').agg({
            'invoicedate': lambda x: (today - x.max()).days,
            'invoiceno': 'nunique',
            'totalamount': 'mean'
        })

        clv_df = clv_df.reset_index()
        clv_df.columns = ['CustomerID', 'Recency', 'Frequency', 'AOV']

        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at: {model_path}")
        else:
            X = clv_df[['Frequency', 'Recency', 'AOV']]
            clv_df['Predicted_CLV'] = model.predict(X)

            result = pd.merge(rfm, clv_df[['CustomerID', 'Predicted_CLV']], on='CustomerID')

            st.success("✅ CLV Prediction Done!")
            st.dataframe(result[['CustomerID', 'Segment', 'Predicted_CLV']].sort_values(by='Predicted_CLV', ascending=False).style.format({"Predicted_CLV": "{:.2f}"}))

            # CLV visualizations
            st.subheader("📊 CLV Analysis")
            fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(result['Predicted_CLV'], bins=30, kde=True, color=primary_color, ax=axes[0])
            axes[0].set_title("CLV Distribution")

            sns.boxplot(x=result['Predicted_CLV'], ax=axes[1], color=secondary_color)
            axes[1].set_title("CLV Boxplot")

            sns.scatterplot(data=result, x='Monetary', y='Predicted_CLV', ax=axes[2], color=primary_color)
            axes[2].set_title("Monetary vs CLV")
            axes[2].set_xlabel("Monetary")
            axes[2].set_ylabel("CLV")
            
            sns.countplot(data=rfm, x='Segment', hue='Segment', order=rfm['Segment'].value_counts().index, palette='Set2', ax=ax1, legend=False)


            st.pyplot(fig2)
            add_download_button(fig2, "Download CLV Analysis")

            # Summary stats
            st.subheader("📊 CLV Summary Stats")
            stats = result['Predicted_CLV'].agg(['mean', 'median', 'max']).rename({
                'mean': 'Mean CLV',
                'median': 'Median CLV',
                'max': 'Max CLV'
            })
            st.table(stats.apply(lambda x: f"${x:,.2f}").to_frame())

            # Download CSV
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full RFM & CLV CSV",
                data=csv,
                file_name='RFM_CLV_Segmented_Customers.csv',
                mime='text/csv'
            )
            print("combined_df.columns:", rfm.columns.tolist())

            # Generate PDF directly with BytesIO objects
            pdf_buffer = generate_pdf_report(rfm, rfm, save_fig_to_buffer(fig1), save_fig_to_buffer(fig2))
            fig1_buf = io.BytesIO()
            fig1.savefig(fig1_buf, format='png')
            fig1_buf.seek(0)

            fig2_buf = io.BytesIO()
            fig2.savefig(fig2_buf, format='png')
            fig2_buf.seek(0)
            
            st.download_button(
                label="📄 Download Full Analysis Report (PDF)",
                data=pdf_buffer,
                file_name="RFM_CLV_Report.pdf",
                mime="application/pdf"
            )
            
            
            required_cols = ['customerid', 'invoiceno', 'invoicedate', 'totalamount']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Convert 'invoicedate' to datetime
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df.dropna(subset=['invoicedate'], inplace=True)

        # RFM Calculation
        snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('customerid').agg({
            'invoicedate': lambda x: (snapshot_date - x.max()).days,
            'invoiceno': 'nunique',
            'totalamount': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # RFM Scoring
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # Segmenting
        seg_map = {
            r'[4][4][4]': 'Best Customers',
            r'[3-4][3-4][3-4]': 'Loyal Customers',
            r'[2-3][2-3][2-3]': 'Potential Loyalists',
            r'[1-2][1-2][1-2]': 'At Risk',
            r'[1][1][1]': 'Lost'
        }

        def assign_segment(score):
            for pattern, segment in seg_map.items():
                if re.fullmatch(pattern, score):
                    return segment
            return 'Others'

        rfm['Segment'] = rfm['RFM_Score'].apply(assign_segment)

        # CLV Prediction using the model
        clv_input = rfm[['Recency', 'Frequency', 'Monetary']]
        rfm['Predicted_CLV'] = model.predict(clv_input)
        combined_df = rfm.copy()

        st.success("✅ RFM analysis and CLV prediction completed.")
        st.dataframe(combined_df.head(10))

        # Visualizations
        fig1, ax1 = plt.subplots()
        sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='set2', ax=ax1)
        ax1.set_title("Customer Segment Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        add_download_button(fig1, "Download Segment Chart")

        st.subheader("💰 Predicted CLV Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(rfm['Predicted_CLV'], bins=20, kde=True, ax=ax2, color=primary_color)
        st.pyplot(fig2)
        add_download_button(fig2, "Download CLV Chart")
        
        fig1_buf = io.BytesIO()
        fig1.savefig(fig1_buf, format='png')
        fig1_buf.seek(0)

        fig2_buf = io.BytesIO()
        fig2.savefig(fig2_buf, format='png')
        fig2_buf.seek(0)

     

            # Generate PDF buffer
        pdf_buffer = generate_pdf_report(rfm, combined_df, save_fig_to_buffer(fig1), save_fig_to_buffer(fig2))

            # Provide download button for PDF
        st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="Customer_Segmentation_CLV_Report.pdf",
                mime="application/pdf"
            )
        st.write("📌 Segment column:", rfm['Segment'].unique())
        st.write("📌 CLV predictions:", rfm['Predicted_CLV'].describe())




    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📁 Please upload a CSV file to start.")

