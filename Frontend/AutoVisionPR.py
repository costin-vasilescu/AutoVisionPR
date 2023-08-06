from inference import InferencePipeline
import streamlit as st
import requests
import os


def logout_clicked():
    st.session_state.pop('auth')


if 'auth' not in st.session_state:
    with st.form('Login'):
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        submit_button = st.form_submit_button(label='Login')

        if submit_button:
            data = {
                'username': email,
                'password': password
            }
            x = requests.post(
                st.secrets['api_route'] + '/token',
                data=data
            )
            if x.status_code == 200:
                st.session_state['auth'] = (email, x.json()['access_token'])
                st.experimental_rerun()
            elif x.status_code == 401:
                st.error(x.json()['detail'])
else:
    st.markdown('Currently logged in as: ' + st.session_state['auth'][0])
    logout = st.button('Logout', on_click=logout_clicked)
    st.write('---')
    st.markdown('# AutoVisionPR')

    if 'inference_pipeline' not in st.session_state:
        try:
            st.session_state['inference_pipeline'] = InferencePipeline()
        except:
            st.write('Inference Pipeline ❌')
            st.error('Failed to load Inference Pipeline')
            st.stop()
    st.write('Inference Pipeline ✔️')

    if not st.session_state.inference_pipeline.yolo_loaded:
        try:
            st.session_state.inference_pipeline.load_yolo('my_dataset_v2.pt')
            st.session_state['yolo_loaded'] = True
        except:
            st.write('YOLOv8 🚀 ❌')
            st.error('Failed to load YOLOv8')
            st.stop()
    st.write('YOLOv8 🚀 ✔️')

    if not st.session_state.inference_pipeline.lanyocr_loaded:
        try:
            st.session_state.inference_pipeline.load_lanyocr()
            st.session_state['lanyocr_loaded'] = True
        except:
            st.write('LanyOCR ❌')
            st.error('Failed to load LanyOCR')
            st.stop()
    st.write('LanyOCR ✔️')

    # Create required directories if they don't exist
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    if not os.path.exists('library'):
        os.mkdir('library')

    # Initialize session state variables
    # Default settings
    st.session_state['show_fps'] = True
    st.session_state['save_to_library'] = False
    st.session_state['save_to_database'] = True

    st.success('Application is ready to use!')
