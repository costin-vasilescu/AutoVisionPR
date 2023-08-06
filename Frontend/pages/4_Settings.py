import streamlit as st
import os

if 'auth' not in st.session_state:
    st.warning("You need to login to view this content")
    st.stop()
elif 'inference_pipeline' not in st.session_state or 'yolo_loaded' not in st.session_state or 'lanyocr_loaded' not in st.session_state:
    st.warning("Please wait while the application is loading...")
    st.stop()

default_model = 'my_dataset_v2.pt'


def show_fps_clicked():
    st.session_state['show_fps'] = not st.session_state['show_fps']


def save_to_library_clicked():
    st.session_state['save_to_library'] = not st.session_state['save_to_library']


def save_to_database_clicked():
    st.session_state['save_to_database'] = not st.session_state['save_to_database']


st.markdown('# Settings')
show_fps = st.checkbox('Show FPS', value=st.session_state.show_fps, on_change=show_fps_clicked)
save_to_library = st.checkbox('Save output to library', value=st.session_state.save_to_library,
                              on_change=save_to_library_clicked)
save_to_database = st.checkbox('Upload information to database', value=st.session_state.save_to_database,
                               on_change=save_to_database_clicked)
st.write('---')

st.markdown('## YOLO')
weights_path = 'D:/AutoVisionPR App/weights'
weights = os.listdir(weights_path)
weight_selection = st.selectbox('Model', weights)
if st.button('Apply'):
    try:
        st.session_state.inference_pipeline.load_yolo(default_model)
    except:
        st.error(f'Failed to load {weight_selection}')
    st.success(f'Succesfully loaded {weight_selection} ️')

# Confidence threshold
if st.checkbox('Confidence Threshold', value=False, key='yolo_enabled'):
    threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='conf_threshold')
    st.session_state.inference_pipeline.conf_threshold = threshold
else:
    st.session_state.inference_pipeline.conf_threshold = 0

# Tracking algorithm
selected_index = 0 if st.session_state.inference_pipeline.tracker == 'botsort.yaml' else 1
tracker = st.radio('Tracking', ['BoTSORT', 'ByteTrack'], index=selected_index)
if tracker == 'BoTSORT':
    st.session_state.inference_pipeline.tracker = 'botsort.yaml'
elif tracker == 'ByteTrack':
    st.session_state.inference_pipeline.tracker = 'bytetrack.yaml'
st.write('---')
