# --- ZORA FUNCTION: Generate AI response for EMS prompts ---
def ask_zora(ems_profile, patwit_profile, prompt):
    """
    Generates a Zora AI response based on the current EMS and PatWit profiles and an EMS prompt.

    IMPORTANT:
    - This function is intentionally lean and ONLY handles the QA logic.
    - It does NOT decide when to call (e.g., no zora_prompt gating here).
    - It does NOT mutate profiles.
    - Pass in whatever profile subsets you want Zora to see (you can pre-trim empties upstream).

    Args:
        ems_profile (dict): The current EMS profile (ideally only populated fields).
        patwit_profile (dict): The current PatWit profile (ideally only populated fields).
        prompt (str): The specific prompt or question from EMS (without any wake word).

    Returns:
        str: AI-generated response from Zora (plain text, suitable for TTS).
    """

    import openai  # Local import to keep this module standalone
    import json
    
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the current working directory
    #openai.api_key = "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"

    # System instruction (kept as-is per your request)
    system_instruction = (
        "You are Zora, an EMS AI assistant helping paramedics make fast, informed decisions."
        "Respond with a clinical impression based on the provided patient info (EMS Profile is fully reliable, Patwit profile is suplimental information)."
        "Be medically accurate and prioritize brevity — give the essential insight first."
        "Keep responses as short as possible unless more explanation is absolutely necessary."
        "If asked to 'summarize patient condition' or something similar, condense patient info and output a short response that prioritizes only essential info."
    )

    # Compose the minimal context: EMS + PatWit profiles as separate sections.
    # (No gating or extra logic here; you decide upstream what to include.)
    context_block = (
        "EMS profile:\n" + json.dumps(ems_profile, indent=2) + "\n\n"
        "PatWit profile:\n" + json.dumps(patwit_profile, indent=2)
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": context_block},
        {"role": "user", "content": prompt.strip()},
    ]

    try:
        # Call OpenAI (model kept as-is per your request)
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        # Extract and clean up response text
        zora_reply = response.choices[0].message.content.strip()

        # Debug print (optional; remove if you don't want console output)
        print(f" Zora AI response: {zora_reply}")

        return zora_reply

    except Exception as e:
        error_message = f"⚠️ Zora encountered an error: {e}"
        print(error_message)
        return error_message
