import streamlit as st
import requests
import pandas as pd
import datetime

if 'auth' not in st.session_state:
    st.warning("You need to login to view this content")
    st.stop()
elif 'inference_pipeline' not in st.session_state or 'yolo_loaded' not in st.session_state or 'lanyocr_loaded' not in st.session_state:
    st.warning("Please wait while the application is loading...")
    st.stop()

# Load sources from database
sources_request = requests.get(
    st.secrets['api_route'] + '/sources'
)
sources = pd.DataFrame(sources_request.json())


source = st.selectbox('Source', [''] + list(sources['name']))
county = st.text_input('County')
identifier = st.text_input('Identifier')
datetime_checked = st.checkbox('Date/Time')
col1, col2 = st.columns(2)
if datetime_checked:
    with col1:
        st.header("From")
        date1 = st.date_input('Select a date', datetime.date.today())
        time1 = st.text_input('Enter a time', value='00:00:00')
    with col2:
        st.header("To")
        date2 = st.date_input('Select another date', datetime.date.today())
        time2 = st.text_input('Enter another time', value='23:59:59')

if st.button(label='Submit'):
    params = {}
    if source:
        source_id = sources[sources['name'] == source]['id'].values[0]
        params['source_id'] = source_id
    if county:
        params['county'] = county
    if identifier:
        params['identifier'] = identifier
    if datetime_checked:
        params['datetime1'] = date1.strftime('%Y-%m-%d') + ' ' + time1
        params['datetime2'] = date2.strftime('%Y-%m-%d') + ' ' + time2

    if params:
        detections_request = requests.get(
            st.secrets['api_route'] + '/detection_search', params=params
        )
    else:
        detections_request = requests.get(
            st.secrets['api_route'] + '/detections'
        )

    detections = pd.DataFrame(detections_request.json())
    if not detections.empty:
        detections = detections.drop('id', axis=1)
    st.dataframe(detections)
