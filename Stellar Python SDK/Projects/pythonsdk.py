import streamlit as st
from stellar_sdk import Server
import pandas as pd

# Setup
server = Server("https://horizon.stellar.org")
st.title("🏦 DMMK Account Dashboard")

# User Input
account_id = st.text_input("Enter Wallet Address (G...)", "")

if account_id:
    try:
        # STEP 1: Fetch transactions specifically for payments
        # We use .payments() instead of .operations() to get clean money-flow data
        response = server.payments().for_account(account_id).limit(50).order(desc=True).call()
        records = response['_embedded']['records']

        # STEP 2: Safety Check - If records is empty, don't run the rest
        if not records:
            st.warning("This account has no transaction history.")
        else:
            money_in = 0.0
            money_out = 0.0
            history = []

            for op in records:
                # Filter for DMMK assets specifically
                asset_code = op.get('asset_code', 'XLM') 
                if asset_code == 'DMMK':
                    amount = float(op['amount'])
                    
                    # Determine Income vs Outcome
                    if op['to'] == account_id:
                        money_in += amount
                        direction = "INCOME"
                    else:
                        money_out += amount
                        direction = "OUTCOME"
                    
                    history.append({
                        "Time": op['created_at'],
                        "Amount": amount,
                        "Type": direction,
                        "From": op['from'][:6] + "...",
                        "To": op['to'][:6] + "..."
                    })

            # STEP 3: The "Money In / Money Out" Visuals
            col1, col2 = st.columns(2)
            col1.metric("Total Money In (DMMK)", f"{money_in:,.2f}")
            col2.metric("Total Money Out (DMMK)", f"{money_out:,.2f}")

            # STEP 4: The Transaction Table (Summed up automatically)
            if history:
                df = pd.DataFrame(history)
                st.subheader("Recent DMMK Transactions")
                st.dataframe(df, use_container_width=True)
                
                # Visual Chart
                st.bar_chart(df.set_index('Time')['Amount'])

    except Exception as e:
        st.error(f"Error fetching data: {e}")
