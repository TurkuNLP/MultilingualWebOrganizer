from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


def get_english_topic_classification_prompt(
    text: str, url: str | None, labels: str, examples: str
) -> list[ChatMessage]:
    """
    Generates a prompt for classifying the topic of an English web page.

    Args:
        text (str): The content of the web page.
        url (str)|None: The URL of the web page if available.
        labels (str): A string representation of the 24 topic labels.
        examples (str): A string representation of example topic classifications.

    Returns:
        list[ChatMessage]: System and user messages for topic classification.
    """

    system_prompt = f"""
  Your task is to classify the topic of a web page into one or more categories.
  Choose the topics from the provided list that best match what the web page content is about. If multiple topics are relevant, order them by importance, with the most prominent topic first.
  Remember to focus on the topic, and not the format, e.g., a book excerpt about a first date is related to 'Social Life' and not 'Literature'.
  {"The URL might help you understand the content. Avoid shortcuts such as word overlap between the page and the topic descriptions or simple patterns in the URL." if url else ""}
  If more than one topic is relevant, return multiple topics in order of importance, with the most important topic first.
  Return topic IDs, not topic names. Never return the same topic ID more than once and never invent a topic ID that is not in the list.
  You must always return at least one topic ID. Never leave the labels array empty.
  Also, return a brief rationale for your classification in the 'rationale' field. The rationale should be a short, concise explanation of why you chose the topics you did, based on the content of the web page.
  Return only a JSON object such as {{"rationale": "...", "labels": ["<labelX>", "<labelY>"]}}. The labels array must contain each relevant topic at most once and must be ordered from most to least important.
"""

    user_prompt = f"""

  Choose the topic(s) from the following list:
  {labels}

  Consider the following web page:

  {f"URL: `{url}`" if url else ""}
  Content: ```
  {text}
  ```
  Examples: ```
  {examples}
  ```
  """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_multilingual_topic_classification_prompt(
    text: str, url: str | None, labels: str, examples: str, language: str
) -> list[ChatMessage]:
    """
    Generates a prompt for classifying the topic of a multilingual web page.

    Args:
        text (str): The content of the web page.
        url (str)|None: The URL of the web page if available.
        labels (str): A string representation of the 24 topic labels.
        examples (str): A string representation of example topic classifications.

    Returns:
        list[ChatMessage]: System and user messages for topic classification.
    """

    system_prompt = f"""
  Your task is to classify the topic of a web page into one or more categories.
  The language of the web page is {language}. Do not let the language of the web page affect your classification. Focus on the content and meaning of the text, rather than the language it is written in.
  Choose the topics from the provided list that best match what the web page content is about. If multiple topics are relevant, order them by importance, with the most prominent topic first.
  Remember to focus on the topic, and not the format, e.g., a book excerpt about a first date is related to 'Social Life' and not 'Literature'.
  {"The URL might help you understand the content. Avoid shortcuts such as word overlap between the page and the topic descriptions or simple patterns in the URL." if url else ""}
  If more than one topic is relevant, return multiple topics in order of importance, with the most important topic first.
  Return topic IDs, not topic names. Never return the same topic ID more than once and never invent a topic ID that is not in the list.
  You must always return at least one topic ID. Never leave the labels array empty.
  Also, return a brief rationale for your classification in the 'rationale' field. The rationale should be a short, concise explanation of why you chose the topics you did, based on the content of the web page.
  Return only a JSON object such as {{"rationale": "...", "labels": ["<labelX>", "<labelY>"]}}. The labels array must contain each relevant topic at most once and must be ordered from most to least important.
"""

    user_prompt = f"""

  Choose the topic(s) from the following list:
  {labels}
 
  Consider the following web page:

  {f"URL: `{url}`" if url else ""}
  Content: ```
  {text}
  ```
  Examples: ```
  {examples}
  ```
  """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
