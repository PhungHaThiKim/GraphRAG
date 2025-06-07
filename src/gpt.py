from openai import OpenAI

# === Global Configuration ===
API_KEY = None  # You must set your OpenAI API key here (or inject from environment)


def simulate_fn_gpt(
    prompt: str,
    correct_answer: str,
    options: list,
    model_name: str = "gpt-3.5-turbo"
) -> tuple:
    """
    Simulate interaction with GPT API to get an answer for a multiple-choice QA prompt.
    Only the answer letter (A-E) is expected from GPT.

    Args:
        prompt (str): Formatted prompt containing the question and choices.
        correct_answer (str): Ground-truth answer (text).
        options (list): List of all possible answer options (in order A→E).
        model_name (str): OpenAI model to use.

    Returns:
        Tuple: (reward: int (1 or 0), predicted_label: str)
    """
    # Load API key and initialize client
    api_key = API_KEY
    client = OpenAI(api_key=api_key)

    # Fixed instruction for GPT
    system_prompt = (
        "You are a multiple-choice QA system.\n"
        "Only respond with the letter corresponding to your chosen answer (e.g., A, B, C, D, or E).\n"
        "Do not include the answer text."
    )

    try:
        print("\n🟢 [DEBUG] Sending to GPT...")
        print("Prompt:\n", prompt)

        # Send request to OpenAI Chat API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1,  # Limit to single letter (A-E)
        )

        # Get GPT predicted answer letter (e.g., "A")
        prediction = response.choices[0].message.content.strip().upper()

        # Map label letter to text answer
        label_map = {chr(65 + i): opt.strip().lower() for i, opt in enumerate(options)}
        correct_text = correct_answer.strip().lower()
        predicted_text = label_map.get(prediction, "")

        # Reward = 1 if predicted_text matches correct
        reward = int(predicted_text == correct_text)

        # Debug output
        print("GPT Answer:", prediction)
        print("Mapped to:", predicted_text)
        print("Correct:", correct_text)
        print("Reward:", reward)

        return reward, prediction

    except Exception as e:
        print("❌ GPT error:", e)
        return 0, "ERROR"
