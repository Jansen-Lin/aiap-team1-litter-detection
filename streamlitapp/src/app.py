import glob
import os
from numpy import double
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly_express as px
import cv2
from PIL import Image


def convert_time(string):
    '''
    function convert time string to msec
    '''
    ftr = [3600, 60, 1]
    start_time = sum([a*b for a, b in zip(ftr, map(int, string.split(':')))])
    return start_time

# Headers
st.set_page_config(page_title="Mini Project", layout="wide")
st.title("Little Duck - Inference Results")

# Find predefined paths
csv_paths = glob.glob("..\\processed" + "/*.csv")
mp4_paths = glob.glob("..\\processed" + "/*.mp4")

# Get user inputs
csv_input = st.selectbox("Input path to csv file:", csv_paths)
mp4_input = st.selectbox("Input path to matching mp4 file:", mp4_paths)
# st.radio("Trying out radio", csv_paths)

st.write("Files in the Folder:")
if csv_input:
    # Read the csv file output from peekingdck into dataframe
    # shows = pd.read_csv("data/demo/demo.csv")  # todo dynamically list the directory
    shows = pd.read_csv(csv_input)

    shows['Time_seconds'] = shows['Time'].apply(convert_time)
    shows['Time_seconds'] = shows['Time_seconds'].apply(
        lambda x: x-shows['Time_seconds'][0])
    start_time = shows['Time_seconds'].iloc[0]


    # display in AG grid the dataframe
    gb = GridOptionsBuilder.from_dataframe(shows)
    gb.configure_pagination()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_column('Time', headerCheckboxSelection=True)
    gridOptions = gb.build()

    data = AgGrid(shows,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED)


    selected_rows = data["selected_rows"]
    selected_rows = pd.DataFrame(selected_rows)
    st.write(selected_rows)


# select the aggrid and display the images in sequence
if len(selected_rows) == 1:
    print(selected_rows)
    frame_time = selected_rows['Time_seconds']
    print(frame_time)
    print("type:", type(selected_rows))

    diff_time = frame_time - start_time
    print(diff_time)

    if mp4_input:
        # video_file = "data/demo/demo.mp4" # todo dynamically read from directory
        video_file = mp4_input
        vidcap = cv2.VideoCapture(video_file)

        type(diff_time)

        vidcap.set(cv2.CAP_PROP_FPS, 10)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, float(diff_time))

        img = []
        for i in range(10):
            img.append(vidcap.read()[1])

        for i in range(len(img)):
            st.image(Image.fromarray(img[i]), use_column_width=False)
