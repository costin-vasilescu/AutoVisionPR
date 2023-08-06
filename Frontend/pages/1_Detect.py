# Ignore IDE error for this import statement, streamlit automatically appends parent directory to sys.path
from inference import OCRThread
from moviepy.editor import VideoFileClip
import requests
import streamlit as st
import time
import cv2
import os

if 'auth' not in st.session_state:
    st.warning("You need to login to view this content")
    st.stop()
elif 'inference_pipeline' not in st.session_state or 'yolo_loaded' not in st.session_state or 'lanyocr_loaded' not in st.session_state:
    st.warning("Please wait while the application is loading...")
    st.stop()


def start_clicked():
    st.session_state.started = True


def stop_clicked():
    st.session_state.started = False


def video_length(path):
    video = cv2.VideoCapture(path)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    video.release()
    return total_frames / fps


if 'started' not in st.session_state:
    st.session_state['started'] = False
elif not st.session_state.started and 'ocr_thread' in st.session_state:
    st.session_state.ocr_thread.stop()

image_extensions = ['.bmp', '.dng', 'jpeg', 'jpg', '.mpo', 'png', '.tif', '.tiff', '.webp', '.pfm']
video_extensions = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', 'mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm']

src = st.radio('Source', ['Upload', 'Stream'])

if src == 'Upload':
    uploaded_file = st.file_uploader('')
    url = None
elif src == 'Stream':
    uploaded_file = None
    url = st.text_input('Enter URL', help='YouTube URL or RTSP/RTMP/HTTP stream URL')

if uploaded_file or url:
    if uploaded_file:
        source = uploaded_file.name
        path = './uploads/' + source
    elif url:
        source = url
        path = None

    if st.session_state.started:
        stframe = st.empty()
        stframe.write('Loading...')
        post_frame = st.empty()
        post_frame.button('Stop', on_click=stop_clicked)

        st.session_state.inference_pipeline.setup(path)
        if 'ocr_thread' in st.session_state:
            if not st.session_state.ocr_thread.is_stop_set():
                st.session_state.ocr_thread.stop()
            st.session_state.ocr_thread.join()

        st.session_state.ocr_thread = OCRThread(st.session_state.inference_pipeline)
        st.session_state.ocr_thread.start()

        # Don't show FPS if the source is an image
        if os.path.splitext(uploaded_file.name)[1].lstrip('.') in image_extensions:
            show_fps = False
        else:
            show_fps = st.session_state.show_fps

        # Save output to library
        if st.session_state.save_to_library:
            video = cv2.VideoCapture(path)

            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video.release()

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            output_path = './library/temp.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        previous_time = time.time()
        for result in st.session_state.inference_pipeline.results:
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            annotated_frame = st.session_state.inference_pipeline.process_frame(result)
            if st.session_state.save_to_library:
                out.write(annotated_frame)
            if show_fps:
                annotated_frame = cv2.putText(
                    annotated_frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3
                )
            stframe.image(annotated_frame, channels='BGR')

        post_frame.write('Waiting for OCR thread to finish')
        st.session_state.ocr_thread.stop()
        st.session_state.ocr_thread.join()

        if st.session_state.save_to_library:
            st.write('Saving output to library')
            out.release()

            # Resave video with MIME type 'video/mp4'.
            video = VideoFileClip(output_path)
            video.write_videofile(f'./library/{os.path.splitext(source)[0]}.mp4', codec="libx264")
            os.remove(output_path)

        # Save to database
        if st.session_state.save_to_database:
            detections_count = len(st.session_state.inference_pipeline.detections_dict)
            st.write(f'Uploading information of {detections_count} detections to database')

            if path is None:
                source_type = 'stream'
            else:
                extension = os.path.splitext(source)[1].lstrip('.')
                source_type = 'image' if extension in image_extensions else 'video'

            source_request = requests.post(
                st.secrets['api_route'] + '/sources',
                json={
                    'name': source,
                    'type': source_type,
                    'length': video_length(path) if source_type == 'video' else None,
                    'detections': detections_count
                }
            )
            if source_request.status_code == 200:
                source_id = source_request.json()['id']
            else:
                get_source = requests.get(st.secrets['api_route'] + f'/source/{source}')
                source_id = get_source.json()['id']

            ocr_dict = st.session_state.inference_pipeline.ocr_dict
            detections_dict = st.session_state.inference_pipeline.detections_dict
            # frame_id, tracking_id, bndbox, confidence, timestamp, plate_image
            # st.write(st.session_state.inference_pipeline.detections_dict)

            for key, value in detections_dict.items():
                if key is None:
                    continue

                # (text, plate_score)
                ocr_output = ocr_dict.get(key, (None, None))
                if ocr_output[0] is not None:
                    text = ocr_output[0]
                    text_score = ocr_output[1]

                    if text[0] == 'B' and not text[1].isupper():
                        county = 'B'
                        identifier = text[1:]
                    else:
                        county = text[:2]
                        identifier = text[2:]
                else:
                    county = None
                    identifier = None
                    text_score = None

                detection = value
                requests.post(
                    st.secrets['api_route'] + '/detections',
                    json={
                        'source_id': str(source_id),
                        'frame_id': str(detection[0]),
                        'tracking_id': str(key),
                        'bndbox': str(detection[1]),
                        'plate_confidence': str(detection[2]),
                        'text_score': str(text_score) if text_score is not None else '',
                        'county': str(county) if county is not None else '',
                        'identifier': str(identifier) if identifier is not None else '',
                        'timestamp': str(detection[3])
                    }
                )

        st.write('Finished')
        st.button('Reset')
        st.session_state.started = False
    else:
        if uploaded_file is not None:
            with open(f'./uploads/{uploaded_file.name}', 'wb+') as f:
                f.write(uploaded_file.getvalue())
        start_button = st.button('Start', on_click=start_clicked)
