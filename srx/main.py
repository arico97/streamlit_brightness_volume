import streamlit as st
from srx.volume_brightness_control import start_magic,cv2

st.title('AI volume and brightness control')
st.write('Press q botton to stop')
clicked=st.button('Click')
if clicked: 
    start_magic()
    if clicked:
        cv2.VideoCapture(0).release()
        cv2.destroyAllWindows()
        st.stop()