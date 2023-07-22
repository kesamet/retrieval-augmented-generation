"""
Utility functions for Streamlit app.
"""
import base64
import json
import pickle
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


def uri_encode_path(path: str, mime: str = "image/png") -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def add_header(path: str) -> None:
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )


def get_pdf_display(pdfbytes: bytes) -> str:
    base64_pdf = base64.b64encode(pdfbytes).decode("utf-8")
    return f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="970" type="application/pdf"></iframe>
        """


def download_button(
    data,
    filename: str,
    label: str,
    pickle_it: bool = False,
    *args,
    **kwargs,
) -> None:
    mime = None
    if pickle_it:
        try:
            data = pickle.dumps(data)
        except pickle.PicklingError as e:
            st.error(e)
            return
    else:
        if isinstance(data, bytes):
            pass
        elif isinstance(data, pd.DataFrame):
            data = data.to_csv(index=False)
            mime = "text/csv"
        else:
            # Try JSON encode for everything else
            data = json.dumps(data, indent=2)

    try:
        # some strings <-> bytes conversions necessary here
        data = data.encode()
    except AttributeError:
        pass

    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=kwargs.get("key"),
        help=kwargs.get("help"),
        on_click=kwargs.get("on_click"),
        args=args,
        kwargs=kwargs,
    )


def logout_button(auth_domain: str) -> str:
    custom_css, button_id = _custom_button_style()
    return (
        custom_css
        + f"""
        <a id="{button_id}" href="https://{auth_domain}/_oauth/logout"
        target="_self">Logout</a><br></br>
        """
    )


def _custom_button_style():
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
            <style>
            #{button_id} {{
                background-color: #FFFFFF;
                color: #262730;
                padding: 0.4em 0.74em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: #DDDDDD;
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: #F63366;
                color: #F63366;
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: #F63366;
                color: white;
            }}
            </style>
        """
    return custom_css, button_id


def adjust_container_width(width: int = 1000) -> None:
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def remove_menu() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def colour_text(notes: str, color: str = "red") -> None:
    st.markdown(
        f"<span style='color: {color}'>{notes}</span>",
        unsafe_allow_html=True,
    )


def local_css(filename: str) -> None:
    with open(filename, "r") as f:
        st.markdown(
            "<style>{}</style>".format(f.read()),
            unsafe_allow_html=True,
        )
