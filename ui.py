import streamlit as st
from app import convert_specimens

st.title("Pathology Formatter")

text = st.text_area("Paste specimens:")

if st.button("Convert"):
    result = convert_specimens(text)
    st.text_area("Output", result, height=300)

