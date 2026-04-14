import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import plotly.express as px
from datetime import datetime
import sqlite3
import pickle 
from sklearn.model_selection import GridSearchCV

# Load the Model
model = joblib.load(r"C:\Users\samee\DS_course\Bankruptcy_Classification\bankruptcy_model.pkl")

# Must be the first Streamlit command 
st.set_page_config(page_title = "BANKRUPTCY PREVENTION", layout = 'wide')

#---------------Database setup 
conn = sqlite3.connect('predictions.db' , check_same_thread = False)
cursor = conn.cursor()

cursor.execute ( """
CREATE TABLE IF NOT EXISTS Predictions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    industrial REAL,
    management REAL,
    financial REAL,
    credibility REAL,
    competitiveness REAL,
    operating_risk REAL,
    total_risk_score REAL,
    probability REAL,
    prediction_text TEXT,
    Risk_level TEXT
)
""")
#------------------ HELPER FUNCTION
def get_risk_level(trs):
    """High / Medium / Low based on Total Risk Score (0–3 range)"""
    if trs >= 2.0:
        return "🔴 High"
    elif trs >= 1.0:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# --------- UI----------------
st.title("BANKRUPTCY CLASSIFICATION DASHBOARD ")
st.divider()

# SIDEBAR 
st.sidebar.title( "📋 Project Info")
st.sidebar.info("""
**Team:** Group 1

**Model:** KNN 
""")

choice = st.sidebar.radio("Navigation", ["Individual Prediction", "Bulk Prediction", "VIEW HISTORY"])


#=========== INDIVIDUAL PREDICTION ==============
if choice == "Individual Prediction":
    st.header("Single Company Risk Assessment")

    # --- Section 1: User Input Interface ---
    # Capturing business parameters using numeric inputs ranging from 0.0 to 1.0
    st.subheader("Enter Company Parameters")
    col1, col2 = st.columns(2)
    with col1:
        i = st.number_input("Industrial Risk", 0.0, 1.0, 0.5, step=0.5)
        m = st.number_input("Management Risk", 0.0, 1.0, 0.5, step=0.5)
        f = st.number_input("Financial Flexibility", 0.0, 1.0, 0.5, step=0.5)
    with col2:
        c = st.number_input("Credibility", 0.0, 1.0, 0.5, step=0.5)
        cp = st.number_input("Competitiveness", 0.0, 1.0, 0.5, step=0.5)
        o = st.number_input("Operating Risk", 0.0, 1.0, 0.5, step=0.5)

    # --- Section 2: Prediction Engine ---
 
  # --- Section 2: Prediction Engine ---
    if st.button("Predict Now"):
        
        # Feature engineering (IMPORTANT)
        total_risk = i + m + o
        stability = f + c + cp

        input_data = [[i, m, f, c, cp, o, total_risk, stability]]


        try:
            # ✅ Pipeline handles scaling internally
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            # Business logic
            trs_val = i + m + o
            risk_status = get_risk_level(trs_val)

            # Prediction text
            p_text = "Bankruptcy" if prediction == 1 else "Non-Bankruptcy"

            # --- Store in DB ---
            cursor.execute("""
                INSERT INTO Predictions (
                    industrial, management, financial, credibility, competitiveness, 
                    operating_risk, total_risk_score, probability, prediction_text, Risk_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (i, m, f, c, cp, o, trs_val, prob*100, p_text, risk_status))

            conn.commit()

            # --- Display ---
            st.success(f"### Prediction: **{p_text}**")
            st.metric("Bankruptcy Probability", f"{prob*100:.2f}%")
    

            # --- Section 4: Results Display ---
            st.success(f"### Analysis Complete! Prediction: **{p_text}**")
            
            # --- Section 5: Strategic Recommendation (MOVED UP FOR VISIBILITY) ---
            st.markdown("### 💡 Strategic Recommendation")
            if p_text == "Bankruptcy":
                st.error("""
                **High Risk Warning:**
                - **Financial:** Improve Financial Flexibility immediately to boost liquidity.
                - **Operations:** Optimize Operating costs to prevent further internal losses.
                - **Strategy:** Conduct a deep review of Management and Credibility strategies.
                """)
            else:
                st.success("""
                **Stable Status:**
                - **Growth:** The company's risk profile is currently healthy.
                - **Market:** Focus on maintaining high Market Competitiveness.
                - **Standards:** Continue following current industrial standards and financial discipline.
                """)

            # --- Section 6: Visualizations (Plotly Gauge) ---
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = trs_val,
                title = {'text': "Risk Score Meter (Max 3.0)"},
                gauge = {
                    'axis': {'range': [None, 3]},
                    'bar': {'color': "red" if p_text == "Bankruptcy" else "green"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "salmon"}]
                }
            ))
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # --- Section 6: History & Data Management ---
    # Show last prediction result banner if available (persists after rerun)
    if 'last_prediction' in st.session_state:
        lp = st.session_state['last_prediction']
        st.info(f"🔔 Last Prediction: **{lp['p_text']}** | Risk Level: {lp['risk_status']} | Score: {lp['trs_val']:.1f}")

    st.markdown("---")
    st.subheader("📊 Historical Logs")

    # Fetching records from the local database for display (always fresh)
    db_data = pd.read_sql("SELECT * FROM Predictions ORDER BY Id DESC", conn)

    if not db_data.empty:
        st.dataframe(db_data, use_container_width=True)
        # Functionality to reset historical data
        if st.button("🗑️ Clear History"):
            cursor.execute("DELETE FROM Predictions")
            conn.commit()
            st.session_state.pop('last_prediction', None)
            st.success("History cleared successfully!")
            st.rerun()
    else:
        st.info("No historical records available.")

#============= BULK PREDICTIONS ============================
elif choice == "Bulk Prediction":
    st.header("Batch Risk Assessment")

    uploaded_file = st.file_uploader("Upload Company Data", type=["csv", "xlsx"])
    
    if uploaded_file:
        import re # Regex for cleaning
        
        # Load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Original Data Preview:")
        st.dataframe(df.head())

        if st.button("Run Batch Analysis"):
            results_list = []
            processed_rows = []

            for index, row in df.iterrows():
                try:
                    #Convert the entire row into a string and extract all numbers.
                    line = " ".join(map(str, row.values))
                    # Regex to find numbers (0, 0.5, 1)
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                    if len(numbers) >= 6:
                        # Convert the first 6 numbers to float
                        i, m, f, c, cp, o = [float(n) for n in numbers[:6]]

                        # Business Logic
                        trs_val = i + m + o
                        risk_status = get_risk_level(trs_val)

                        # ✅ FIX: Use actual model prediction (same as Individual tab)
                        total_risk = i + m + o
                        stability = f + c + cp
                        input_data = [[i, m, f, c, cp, o, total_risk, stability]]
                        
                        prediction = model.predict(input_data)[0]
                        prob = model.predict_proba(input_data)[0][1]
                        p_text = "Bankruptcy" if prediction == 1 else "Non-Bankruptcy"
                        p_val = (trs_val / 3.0) * 100
                        
                        # Database Insert
                        cursor.execute("""
                            INSERT INTO Predictions (
                                industrial, management, financial, credibility, competitiveness, 
                                operating_risk, total_risk_score, probability, prediction_text, Risk_level
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (i, m, f, c, cp, o, trs_val, p_val, p_text, risk_status))
                        
                        results_list.append([p_text, risk_status])
                        processed_rows.append([i, m, f, c, cp, o])
                
                except Exception:
                    continue

            conn.commit()

            if results_list:
                # Build clean dataframe for display
                res_df = pd.DataFrame(results_list, columns=['Prediction', 'Risk Level'])
                data_df = pd.DataFrame(processed_rows, columns=['Ind', 'Mgmt', 'Fin', 'Cred', 'Comp', 'Oper'])
                final_output = pd.concat([data_df, res_df], axis=1)

                st.success(f"Successfully analyzed {len(final_output)} records!")
                st.dataframe(final_output)

                # Charts
                fig_pie = px.pie(final_output, names='Risk Level', hole=0.4, title="Analysis Summary")
                st.plotly_chart(fig_pie)
            else:
                st.error("Still no numeric data found. Please ensure your file has values like 0, 0.5, or 1.")
#========== HISTORY VIEW
elif choice == "VIEW HISTORY":
    st.header("📜 Prediction History")

    # --- Step 1: Attempt to Fetch Data from SQLite ---
    try:
        # Fetching all records, sorted by the most recent first
        query = "SELECT * FROM Predictions ORDER BY Id DESC"
        history_df = pd.read_sql(query, conn)

        if not history_df.empty:
            # --- Step 2: Show Total Record Count ---
            st.info(f"Total historical records found: {len(history_df)}")

            # --- Step 3: Display the Data Table ---
            st.dataframe(history_df, use_container_width=True)

            # --- Step 4: Add a Download Button ---
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full History as CSV",
                data=csv,
                file_name="Bankruptcy_History_Report.csv",
                mime="text/csv"
            )

            # --- Step 5: Clear History Button ---
            st.markdown("---")
            if st.button("🗑️ Clear All History"):
                cursor.execute("DELETE FROM Predictions")
                conn.commit()
                st.success("All records have been deleted successfully!")
                st.rerun()  # Refresh page to show empty state

        else:
            # This shows if the table exists but has 0 rows
            st.warning("No history found. Please perform some predictions in the 'Individual' or 'Bulk' tabs first.")

    except Exception as e:
        # This triggers if the 'Predictions' table hasn't been created yet
        st.error("The history table does not exist yet. Please run at least one prediction to initialize the database.")
        st.info(f"Technical Detail: {e}")
# --- Footer ---
st.divider()
st.write(f"© {datetime.now().year} | Developed by Group 1")