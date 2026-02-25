import streamlit as st
from stellar_sdk import Server
import pandas as pd

# --- Config ---
HORIZON_URL = "https://horizon.stellar.org"
ASSET_CODE = "DMMK"
DMMK_ISSUER = None  # e.g., "G...ISSUER_PUBLIC_KEY". Set to None if unknown.

# --- Streamlit UI ---
st.set_page_config(page_title="DMMK Monitor", layout="wide")
st.title("📊 DMMK Transaction Dashboard")

account_id = st.text_input("Paste Stellar Public Key (starts with G):", "")

if account_id:
    try:
        server = Server(HORIZON_URL)

        # Fetch recent operations for the account (payments come as operations)
        # You could also use server.payments().for_account(account_id) to narrow to payments only.
        resp = server.operations().for_account(account_id).limit(100).order(desc=True).call()
        records = resp.get('_embedded', {}).get('records', [])

        if not records:
            st.warning("⚠️ This account has no recorded transactions on the Stellar Network.")
        else:
            income_total = 0.0
            outcome_total = 0.0
            tx_list = []

            for op in records:
                # Only consider pure payment ops. If you want to include path payments, add their types.
                if op.get('type') != 'payment':
                    continue

                # Non-native assets carry asset_code + asset_issuer
                op_asset_code = op.get('asset_code')
                op_asset_issuer = op.get('asset_issuer')

                if op_asset_code != ASSET_CODE:
                    continue
                if DMMK_ISSUER and (op_asset_issuer != DMMK_ISSUER):
                    continue

                # Amount and direction
                try:
                    amt = float(op.get('amount', '0'))
                except ValueError:
                    continue

                # If 'to' is our account, it's incoming; else outgoing
                is_incoming = (op.get('to') == account_id)
                flow_type = "📥 INCOME" if is_incoming else "📤 OUTCOME"
                if is_incoming:
                    income_total += amt
                else:
                    outcome_total += amt

                tx_list.append({
                    "Timestamp": op.get('created_at', ''),
                    "Amount": amt,
                    "Type": flow_type,
                    "From": op.get('from', ''),
                    "To": op.get('to', ''),
                    "Asset Issuer": op_asset_issuer or '',
                    "Tx Hash": (op.get('transaction_hash') or '')[:12] + "…" if op.get('transaction_hash') else ''
                })

            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Money IN", f"{income_total:,.2f} {ASSET_CODE}")
            with col2:
                st.metric("Total Money OUT", f"{outcome_total:,.2f} {ASSET_CODE}")
            with col3:
                net = income_total - outcome_total
                st.metric("Net Balance", f"{net:,.2f} {ASSET_CODE}", delta=net)

            # --- Table + Chart ---
            if tx_list:
                df = pd.DataFrame(tx_list)
                st.subheader("Recent Activity Details")
                st.dataframe(df, use_container_width=True)
                try:
                    # Bar chart by time — aggregate in case of duplicate timestamps
                    chart_df = (
                        df.groupby("Timestamp", as_index=True)["Amount"]
                        .sum()
                        .sort_index()
                    )
                    st.bar_chart(chart_df)
                except Exception:
                    st.info("Chart not available (timestamp parsing issue).")
            else:
                st.info(f"Account exists, but no {ASSET_CODE}-specific payments found.")
    except Exception as e:
        st.error(f"Error: {e}. Please check if the Public Key is valid and that dependencies are installed.")
