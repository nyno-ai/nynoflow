from nynoflow.chats._chatgpt._chatgpt_objects import ChatgptResponse


def render_chatgpt_response(response_content: str) -> ChatgptResponse:
    """Render a ChatGPT response based on the given value. The token counts will be falsy.

    Args:
        response_content (str): The value to put in the assistant response.

    Returns:
        ChatgptResponse: The rendered ChatGPT response.
    """
    chatgpt_response: ChatgptResponse = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{response_content}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    return chatgpt_response
