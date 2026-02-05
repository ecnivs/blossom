import webbrowser


def browser_handler(url: str):
    try:
        webbrowser.open(url)
        return "Opened URL: " + url
    except Exception as e:
        return f"Failed to open URL: {e}"


def register():
    return {
        "name": "browser_tool",
        "description": "Opens a URL in the default web browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open"},
            },
            "required": ["url"],
        },
        "handler": browser_handler,
        "direct_tts": True,
    }
