import json
import requests
from google import genai
import time

# --- CONFIGURATION ---
DECK_NAME = input('Enter deck name: ')
API_KEY = "AIzaSyBwcnJfmdfL0G4UnhtOwX8z7llwh4ThWK0"
JP_FIELD = "SentenceFront"
EN_FIELD = "SentenceBack"
KEYWORDS_FIELD = "Meaning"
# IMPORTANT: Paste your NEW API Key here. Never share it publicly!
# API_KEY = "API"
# ---------------------

def anki_request(action, **params):
    """Sends a request to the local AnkiConnect API."""
    payload = {'action': action, 'params': params, 'version': 6}
    response = requests.post('http://localhost:8765', json=payload).json()
    if response.get('error'):
        raise Exception(f"AnkiConnect Error: {response['error']}")
    return response.get('result')

def main():
    print(f"Finding notes in deck: {DECK_NAME}...")

    # 1. Get all note IDs in the specified deck
    query = f'deck:"*{DECK_NAME}*"'
    try:
        note_ids = anki_request('findNotes', query=query)
    except Exception as e:
        print(f"Failed to connect to Anki or find deck: {e}")
        return

    # Store the total number of notes for our progress tracker
    total_notes = len(note_ids)
    if total_notes == 0:
        print("No notes found. Please check your DECK_NAME spelling.")
        return

    print(f"Found {total_notes} notes. Fetching data...")

    # 2. Get the actual content of those notes
    notes_info = anki_request('notesInfo', notes=note_ids)

    # Initialize the Gemini API client
    client = genai.Client(api_key=API_KEY)

    # 3. Loop through notes, translate, extract keywords, and update
    for index, note in enumerate(notes_info, 1):
        note_id = note['noteId']
        fields = note['fields']

        # Check if ALL required fields exist on this note
        if JP_FIELD not in fields or EN_FIELD not in fields or KEYWORDS_FIELD not in fields:
            print(f"[{index}/{total_notes}] Skipping Note {note_id}: One or more target fields not found in this Anki note type.")
            continue

        jp_text = fields[JP_FIELD]['value']

        if jp_text.strip():
            print(f"\n[{index}/{total_notes}] Processing: {jp_text}")

            prompt = f"""
                        Analyze the following Japanese text: {jp_text}

                        Provide the output in EXACTLY this format. Do not use conversational filler:

                        Meaning:
                        [Insert the full English translation of the sentence here]

                        KeyWords:
                        [Insert Underlined Word's English translation]
                        ------------------------------------------------------------
                        [Word] ([Hiragana Reading]) — [English Meaning or Grammar function]
                        [Repeat for all key vocabulary]
                        Do Not Use Romaji!
                        """

            # --- SMART RETRY SYSTEM ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt
                    )

                    raw_text = response.text.strip()
                    print(f"Result:\n{raw_text}\n{'-'*30}")

                    # --- SPLIT THE TEXT INTO TWO FIELDS ---
                    if "KeyWords:" in raw_text:
                        parts = raw_text.split("KeyWords:")

                        # Clean up the Meaning half
                        meaning_part = parts[0].replace("Meaning:", "").strip()
                        meaning_formatted = meaning_part.replace('\n', '<br>')

                        # Clean up the Keywords half
                        keywords_part = parts[1].strip()
                        keywords_formatted = keywords_part.replace('\n', '<br>')
                    else:
                        # Fallback if the model misses the formatting instructions
                        meaning_formatted = raw_text.replace('\n', '<br>')
                        keywords_formatted = ""

                    # Update the note in Anki with BOTH fields
                    anki_request('updateNoteFields', note={
                        'id': note_id,
                        'fields': {
                            EN_FIELD: meaning_formatted,
                            KEYWORDS_FIELD: keywords_formatted
                        }
                    })

                    time.sleep(2)
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        print(f"⚠️ Rate limit hit! Waiting 30 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(30)
                    else:
                        print(f"[{index}/{total_notes}] Failed to translate note {note_id}: {e}")
                        break

    print("\nProcessing complete!")

if __name__ == '__main__':
    main()