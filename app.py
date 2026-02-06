import streamlit as st
import pandas as pd
from gemini import load_json_response_schema
from gemini import conn as gemini_conn
from PIL import Image
from models import InputType, DataColumns
from media_topics import media_topics_labels
from connection import conn


def app():

    st.title("Auto-Tagging CV")

    st.write('"Auto-tag" content using the IPTC Media Topics controlled vocabulary.')

    toggle = st.radio(
        label="Input Type",
        options=[InputType.TEXT.value, InputType.IMAGE.value],
        horizontal=True,
        label_visibility="hidden"
    )

    if toggle == InputType.TEXT.value:

        with st.form("text_input_form", border=False):

            st.session_state['text_input'] = st.text_area(
                label="Text Input",
                label_visibility="collapsed"
            )
            st.form_submit_button("Submit")

    elif toggle == InputType.IMAGE.value:
        file = st.file_uploader(
            label="upload",
            label_visibility="hidden",
            accept_multiple_files=False,
            type=["jpg", "jpeg", "png"],
        )

        if file:
            # Open the image, resize, and display
            img = Image.open(file)
            w, h = img.size
            resize_factor = max(w, h) / 800
            img = img.resize(size=(int(w // resize_factor), int(h // resize_factor)))
            # display the image
            st.image(img)
            # Generate an image caption
            image_caption = gemini_conn.generate_image_caption(image=img)
            # Display the AI generated caption
            st.subheader("AI generated caption")
            st.write(image_caption.caption)
            # Save the caption in the session state as our text input
            st.session_state['text_input'] = image_caption.caption

    if text_input := st.session_state.get('text_input'):
        # query the db with our text
        df = conn.query(query_texts=text_input)

        if isinstance(df, pd.DataFrame):

            # display the relevant query results retrieved from the vector db
            st.subheader("Relevant concepts")
            st.write(f"""Retrieved {df.shape[0]} relevant concepts from the vocabulary.""")
            st.write(f"""**"{df.iloc[0].Concept}"** is the top concept.""")
            st.dataframe(df)
            # classify the text input using the retrieved vocabulary and structured outputs
            # keywords = ['environment', 'environmental pollution', 'environmental clean-up',
            # 'environmental policy', 'government policy', 'politics and government']
            keywords = gemini_conn.classify_media_topics(
                content=text_input,
                response_schema=load_json_response_schema(df[DataColumns.CONCEPT.value].to_list()),
                vocabulary_json=df[[DataColumns.CONCEPT.value, DataColumns.DEFINITION.value]].to_json(orient="records")
            )
            # display the "auto-tagged" keywords
            st.subheader('AI "auto-tags"')
            st.write(f"""AI tagged the content with {len(keywords)} keywords.""")
            options = st.multiselect(
                label="AI Keywords",
                label_visibility="hidden",
                options=media_topics_labels,
                default=keywords
            )
            st.dataframe(options)
        else:
            st.warning("Text input is too short, provide at least 15 words to continue.")


# Run the app
app()
