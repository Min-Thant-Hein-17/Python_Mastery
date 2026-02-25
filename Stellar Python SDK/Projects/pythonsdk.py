import streamlit as st
from stellar_sdk import Server
import pandas as pd

st.set_page_config(page_title="DMMK Fraud Monitor", layout="wide")
server = Server("https://horizon.stellar.org")

st.title("🛡️ DMMK Transaction Dashboard")

# User Input
target_account = st.text_input("Enter Wallet Address (G...)", "")

if target_account:
    try:
        # Fetching data from Stellar
        ops = server.operations().for_account(target_account).limit(50).order(desc=True).call()
        records = ops['_embedded']['records']

        if not records:
            st.warning("No transactions found for this account.")
        else:
            income_total = 0.0
            outcome_total = 0.0
            rows = []

            for op in records:
                # Check for DMMK payments
                if op['type'] == 'payment' and op.get('asset_code') == 'DMMK':
                    amount = float(op['amount'])
                    
                    if op['to'] == target_account:
                        income_total += amount
                        flow = "INCOME"
                    else:
                        outcome_total += amount
                        flow = "OUTCOME"

                    rows.append({
                        "Time": op['created_at'],
                        "Amount": amount,
                        "Flow": flow,
                        "Hash": op['transaction_hash'][:8] + "..."
                    })

            # --- DISPLAY SECTION ---
            # 1. Show the Sums (Metrics)
            col1, col2 = st.columns(2)
            col1.metric("Total Money In (DMMK)", f"{income_total:,.2f}")
            col2.metric("Total Money Out (DMMK)", f"{outcome_total:,.2f}")

            # 2. Show the Individual Transactions
            if rows:
                df = pd.DataFrame(rows)
                st.subheader("Transaction History")
                st.dataframe(df, use_container_width=True)
                
                # 3. Simple Bar Chart
                st.bar_chart(df.set_index('Time')['Amount'])

    except Exception as e:
        st.error(f"Account not found or connection error: {e}")
