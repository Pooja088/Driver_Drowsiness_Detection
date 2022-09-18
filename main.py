import streamlit as st
from processing import *

st.title('Driver Drowsiness')
url = st.text_input("Enter Url")
if st.button("Detect"):
    results = find_drowsy(url)
    for result in results:
        st.image(result)
