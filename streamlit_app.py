import streamlit as st

upload_page = st.Page(
    "receipt_upload.py", title="Upload Receipt", icon=":material/add_circle:"
)
insights_page = st.Page("insights.py", title="Insights", icon=":material/insights:")

pg = st.navigation([insights_page, upload_page])
st.set_page_config(
    page_title="Spending Habits Analyzer",
    page_icon=":material/query_stats:",
    layout="wide",
)
st.title("Spending Habits Analyzer")
# For newline
st.write("\n")

pg.run()
