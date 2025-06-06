"""This is a demo of the llmshield library."""

from openai import OpenAI

import llmshield

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
if not OPENAI_API_KEY:
    msg = "OPENAI_API_KEY is not set!"
    raise ValueError(msg)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def llm_func_wrapper(model, message, temperature, stream):
    # convet message into a list format since history is not yet supported
    messages = [{"role": "user", "content": message}]
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream,
    )


shield = llmshield.LLMShield(llm_func=llm_func_wrapper)


def main() -> None:
    message = (
        "Summarise the letter in a sentence:"
        "Dear John,"
        "I am writing to you to inform you that I will be out of the office "
        "from Monday to Wednesday next week. "
        "I will be available for urgent matters during this time, but please"
        " contact me if you need anything else using "
        "my email address: janesmith@mail.com. "
        "Thank you for your understanding."
        "Best regards,"
        "Jane Smith"
    )

    response = shield.ask(
        model="gpt-4o-mini",
        message=message,
        temperature=0,
        stream=True,
    )

    # skipcq: PTC-W0047
    for _chunk in response:
        pass


if __name__ == "__main__":
    main()
