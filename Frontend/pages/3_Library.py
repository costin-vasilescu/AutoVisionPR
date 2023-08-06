import os
import streamlit as st
import requests
import pandas as pd

if 'auth' not in st.session_state:
    st.warning("You need to login to view this content")
    st.stop()
elif 'inference_pipeline' not in st.session_state or 'yolo_loaded' not in st.session_state or 'lanyocr_loaded' not in st.session_state:
    st.warning("Please wait while the application is loading...")
    st.stop()

sources_request = requests.get(
    st.secrets['api_route'] + '/sources'
)
sources = pd.DataFrame(sources_request.json())

if not sources.empty:
    st.markdown('# Sources')
    sources = sources.drop('id', axis=1)
    st.dataframe(sources)

    # Parse library videos
    st.markdown('# Output')
    library_videos = os.listdir('./library')
    selected_video = st.selectbox('', library_videos)
    if st.button('Load'):
        st.video('./library/' + selected_video)
else:
    st.warning('No sources found')
