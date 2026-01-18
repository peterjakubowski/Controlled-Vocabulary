import streamlit as st
from db import database
import pandas as pd
from gemini import load_json_response_schema, classify_media_topics


def app():

    st.title("IPTC Media Topics CV")

    text_input = st.text_area(
        label="Text Input",
        key="text_input"
    )

    if text_input:
        df = pd.DataFrame(
            data=database.query(
                query_texts=text_input
            ),
            columns=['Concept', 'Count', 'Definition']
        )
        st.subheader("Broad concepts")
        st.dataframe(df)
        response = classify_media_topics(
            content=text_input,
            response_schema=load_json_response_schema(df['Concept'].to_list()),
            vocabulary_json=df[['Concept', 'Definition']].to_json(orient="records")
        )
        st.subheader("AI concepts")
        st.write(pd.DataFrame(response.parsed))


# Run the app
app()
