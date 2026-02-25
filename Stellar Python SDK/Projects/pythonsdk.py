import streamlit as st
from stellar_sdk import Server
import pandas as pd

st.set_page_config(page_title="DMMK Tracker", layout="wide")
server = Server("https://horizon.stellar.org")

st.title("📊 DMMK Transaction Monitor")

account_id = st.text_input("Enter Stellar Account ID (G...)", "")

if account_id:
    try:
        # Fetching operations
        ops = server.operations().for_account(account_id).limit(50).order(desc=True).call()
        records = ops['_embedded']['records']

        income = 0.0
        outcome = 0.0
        tx_data = []

        for op in records:
            if op['type'] == 'payment' and op['asset_code'] == 'DMMK':
                amt = float(op['amount'])
                if op['to'] == account_id:
                    income += amt
                    direction = "📥 Income"
                else:
                    outcome += amt
                    direction = "📤 Outcome"
                
                tx_data.append({
                    "Date": op['created_at'],
                    "Amount": amt,
                    "Type": direction,
                    "Hash": op['transaction_hash'][:10] + "..."
                })

        # --- Dashboard Visualizations ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Income (DMMK)", f"{income:,.2f}")
        with col2:
            st.metric("Total Outcome (DMMK)", f"{outcome:,.2f}")
        with col3:
            net_balance = income - outcome
            st.metric("Net Flow", f"{net_balance:,.2f}", delta=net_balance)

        # --- Chart ---
        if tx_data:
            df = pd.DataFrame(tx_data)
            st.subheader("Transaction History")
            st.dataframe(df, use_container_width=True)
            
            # Simple Bar Chart for Visualization
            st.bar_chart(df.set_index('Date')['Amount'])
        else:
            st.info("No DMMK transactions found for this account.")

    except Exception as e:
        st.error(f"Error: {e}")
