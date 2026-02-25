import streamlit as st

st.title("🛡️ DMMK Fraud Detection Dashboard")

# User Input for Analysis
account_to_check = st.text_input("Enter Stellar Wallet Address (G...)")

if st.button("Analyze Account"):
    with st.spinner('Fetching On-Chain Truth...'):
        results = detect_fraud(account_to_check)
        
        if results:
            st.error(f"Potential Fraud Detected: {len(results)} alerts found!")
            st.table(results)
        else:
            st.success("No immediate bypass patterns detected.")

# Statistics Section 
st.sidebar.header("Key Statistics")
st.sidebar.metric("DMMK/USDT Peg", "1:1", "0.00%")
st.sidebar.metric("Avg. Time-to-Exit", "4.2s", "-1.2s")