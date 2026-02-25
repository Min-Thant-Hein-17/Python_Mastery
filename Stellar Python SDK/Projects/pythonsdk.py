import streamlit as st
from stellar_sdk import Server
import pandas as pd

# Initialization
server = Server("https://horizon.stellar.org")
st.set_page_config(page_title="DMMK Monitor", layout="wide")

st.title("📊 DMMK Transaction Dashboard")

# User input for the single account
account_id = st.text_input("Paste Stellar Public Key (starts with G):", "")

if account_id:
    try:
        # Fetch operations
        # Use .call() to get the actual data
        response = server.operations().for_account(account_id).limit(50).order(desc=True).call()
        records = response['_embedded']['records']

        # SAFETY CHECK: If no transactions exist, 'records' is empty
        if not records:
            st.warning("⚠️ This account has no recorded transactions on the Stellar Network.")
        else:
            income_total = 0.0
            outcome_total = 0.0
            tx_list = []

            for op in records:
                # Filter for DMMK payments only
                if op['type'] == 'payment' and op.get('asset_code') == 'DMMK':
                    amt = float(op['amount'])
                    
                    # Logic: If 'to' is this account, it's Money In.
                    if op['to'] == account_id:
                        income_total += amt
                        flow_type = "📥 INCOME"
                    else:
                        outcome_total += amt
                        flow_type = "📤 OUTCOME"
                    
                    tx_list.append({
                        "Timestamp": op['created_at'],
                        "Amount": amt,
                        "Type": flow_type,
                        "Hash": op['transaction_hash'][:10] + "..."
                    })

            # --- UI: The "Money In / Money Out" Visuals ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Money IN", f"{income_total:,.2f} DMMK")
            with col2:
                st.metric("Total Money OUT", f"{outcome_total:,.2f} DMMK")
            with col3:
                net = income_total - outcome_total
                st.metric("Net Balance", f"{net:,.2f} DMMK", delta=net)

            # --- UI: The Transactions Table ---
            if tx_list:
                df = pd.DataFrame(tx_list)
                st.subheader("Recent Activity Details")
                st.dataframe(df, use_container_width=True)
                
                # Visual Chart
                st.bar_chart(df.set_index('Timestamp')['Amount'])
            else:
                st.info("Account exists, but no DMMK-specific payments found.")

    except Exception as e:
        st.error(f"Error: {e}. Please check if the Public Key is valid.")
