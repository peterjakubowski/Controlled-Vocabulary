import streamlit as st
from db import database
import pandas as pd
from gemini import load_json_response_schema, classify_media_topics, generate_image_caption
from PIL import Image
from models import InputType, DataColumns


def app():

    st.title("IPTC Media Topics CV")

    toggle = st.radio(
        label="Input Type",
        options=[InputType.TEXT.value, InputType.IMAGE.value],
        horizontal=True,
        label_visibility="hidden"
    )

    if toggle == InputType.TEXT.value:

        st.session_state['text_input'] = st.text_area(
            label="Text Input",
        )

    elif toggle == InputType.IMAGE.value:
        file = st.file_uploader(
            label="upload",
            label_visibility="hidden",
            accept_multiple_files=False,
            type=["jpg", "jpeg", "png"],
        )

        if file:
            # Open the image, resize, and display
            img = Image.open(file)  # Open the image
            w, h = img.size
            resize_factor = max(w, h) / 800
            img = img.resize(size=(int(w // resize_factor), int(h // resize_factor)))
            st.image(img)
            image_caption = generate_image_caption(image=img)
            st.session_state['text_input'] = image_caption.caption

    if text_input := st.session_state.get('text_input'):
        # query the db with our text
        query_results = database.query(query_texts=text_input)
        # load the results with occurrence counts and definitions
        df = pd.DataFrame(
            data=query_results,
            columns=[DataColumns.CONCEPT.value, DataColumns.COUNT.value, DataColumns.DEFINITION.value]
        )
        # display the relevant query results
        st.subheader("Broad concepts")
        st.dataframe(df)
        #
        response = classify_media_topics(
            content=text_input if toggle == InputType.TEXT.value else img,
            response_schema=load_json_response_schema(df[DataColumns.CONCEPT.value].to_list()),
            vocabulary_json=df[[DataColumns.CONCEPT.value, DataColumns.DEFINITION.value]].to_json(orient="records")
        )
        st.subheader("AI concepts")
        st.write(pd.DataFrame(response.parsed))


# Run the app
app()
