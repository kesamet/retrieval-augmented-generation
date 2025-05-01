import re


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text.
    Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else text


def replace_special(text: str) -> str:
    return text.replace("$", r"\$")
