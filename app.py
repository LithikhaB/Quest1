"""Streamlit frontend for the dialogue frame locator pipeline."""

import streamlit as st

from src.output.formatter import format_timestamp
from src.pipeline import locate_exact_frame

st.set_page_config(page_title="Dialogue Frame Locator", page_icon=":movie_camera:")
st.title("Dialogue Frame Locator")
st.caption(
    "Finds the exact video frame where a line of dialogue is spoken: "
    "download -> transcribe -> locate -> extract frame."
)

video_url = st.text_input("Video URL", placeholder="https://ok.ru/video/248244667877")
target_dialogue = st.text_input("Target dialogue", placeholder="my mind rebels at stagnation")

if st.button("Run", type="primary", disabled=not (video_url and target_dialogue)):
    try:
        with st.spinner("Downloading, transcribing, and locating..."):
            result = locate_exact_frame(video_url, target_dialogue)
    except Exception as err:
        st.error(f"Pipeline failed: {err}")
    else:
        st.success("Done")
        col1, col2 = st.columns(2)
        col1.metric("Timestamp", format_timestamp(result.timestamp_seconds))
        col2.metric("Frame", result.frame_index)
        st.metric("Confidence", f"{result.confidence:.2f}")
        if result.is_ambiguous:
            st.warning("Ambiguous: audio match confidence is below the threshold.")
        st.markdown(f'**Matched transcript text:** "{result.matched_text}"')
        st.image(str(result.image_path), caption=f"Frame {result.frame_index}")
else:
    st.info("Enter a video URL and the dialogue to search for, then press Run.")
