import streamlit as st
from app import convert_specimens

st.title("Pathology Formatter")

text = st.text_area("Paste specimens:")

if st.button("Convert"):
    result = convert_specimens(text)
    review_lines = [line for line in result.splitlines() if "ERROR - REVIEW REQUIRED" in line]
    warning_lines = [line for line in result.splitlines() if "CHECK SPECIMEN DESCRIPTION" in line]

    if review_lines:
        st.error(
            "Some specimens could not be classified safely and need manual review.",
            icon="🚨",
        )

    if warning_lines and not review_lines:
        st.warning(
            "Some specimens may need manual review.",
            icon="⚠️",
        )

    st.text_area("Output", result, height=300)
