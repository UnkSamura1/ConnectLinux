import json
import urllib.request
import requests
from bs4 import BeautifulSoup
import base64
import time
import re
from pypdf import PdfReader

# --- Configuration ---
DECK_NAME = "Oxford 5k"
MODEL_NAME = "Basic"
MAX_WORDS_TO_TEST = 9999  # Change to 9999 when ready to import the full list

PDF_PATHS = [
    "./Data/The_Oxford_3000_by_CEFR_level.pdf",
    "./Data/The_Oxford_5000_by_CEFR_level.pdf"
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}


# ─────────────────────────────────────────────
# 1. PDF PARSER
# ─────────────────────────────────────────────
def extract_advanced_words(pdf_paths: list[str]) -> dict[str, str]:
    words: dict[str, str] = {}
    entry_re = re.compile(
        r"^([a-zA-Z][a-zA-Z0-9'\-]*(?:\s[a-zA-Z][a-zA-Z0-9'\-]*)?)\s+"
        r"(?:v\.|n\.|adj\.|adv\.|prep\.|conj\.|pron\.|number|interj\.|det\.|auxiliary\.)"
    )
    target_levels = {"B2", "C1"}

    for pdf_path in pdf_paths:
        print(f"Reading PDF: {pdf_path}...")
        try:
            reader = PdfReader(pdf_path)
        except FileNotFoundError:
            print(f"  -> Error: Could not find '{pdf_path}'. Skipping.")
            continue

        current_level = None
        for page in reader.pages:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line: continue

                if line in {"A1", "A2", "B1", "B2", "C1"}:
                    current_level = line
                    continue

                for marker in {"A1", "A2", "B1", "B2", "C1"}:
                    if line.startswith(marker + " ") or line.startswith(marker + "\t"):
                        current_level = marker
                        line = line[len(marker):].strip()
                        break

                if current_level in target_levels and entry_re.match(line):
                    word_part = \
                    re.split(r"\s+(?:v\.|n\.|adj\.|adv\.|prep\.|conj\.|pron\.|interj\.|det\.|auxiliary\.)", line)[
                        0].strip()
                    clean_word = re.sub(r"\d+$", "", word_part).lower()
                    if clean_word not in words:
                        words[clean_word] = current_level

    return words


# ─────────────────────────────────────────────
# 2. ANKICONNECT HELPERS
# ─────────────────────────────────────────────
def invoke(action, **params):
    payload = json.dumps({'action': action, 'params': params, 'version': 6}).encode('utf-8')
    try:
        response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8765', payload)))
    except Exception as e:
        raise Exception(f"Failed to connect to AnkiConnect. Is Anki running? Error: {e}")
    if response.get('error') is not None:
        raise Exception(response['error'])
    return response['result']


def word_exists_in_anki(word: str) -> bool:
    query = f'"deck:{DECK_NAME}" "Front:{word}"'
    existing_notes = invoke('findNotes', query=query)
    return len(existing_notes) > 0


# ─────────────────────────────────────────────
# 3. OXFORD DICTIONARY SCRAPER (WITH EXAMPLES)
# ─────────────────────────────────────────────
def fetch_oxford_data(word: str):
    url = f"https://www.oxfordlearnersdictionaries.com/definition/english/{word}"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return None, None

    soup = BeautifulSoup(res.content, 'html.parser')
    definitions = []

    # 1. Look for individual senses/definitions
    senses = soup.find_all('li', class_='sense')
    for i, sense in enumerate(senses, 1):
        def_el = sense.find('span', class_='def')

        if def_el:
            def_text = def_el.text.strip()

            # 2. Look for the FIRST example sentence under this definition
            # Oxford puts examples inside <span class="x">
            example_el = sense.find('span', class_='x')

            if example_el:
                example_text = example_el.text.strip()
                # Format: Definition followed by indented bullet point example
                definitions.append(f"<b>{i}.</b> {def_text}<br>&nbsp;&nbsp;&nbsp;&nbsp;• <i>{example_text}</i><br>")
            else:
                definitions.append(f"<b>{i}.</b> {def_text}<br>")

    # Fallback if page structure is unusual
    if not definitions:
        for i, span in enumerate(soup.find_all('span', class_='def'), 1):
            definitions.append(f"<b>{i}.</b> {span.text.strip()}<br>")

    if not definitions:
        return None, None

    definition_string = "<br>".join(definitions)

    # Grab Audio
    audio_url = None
    us_pron = soup.find('div', class_='pron-us')
    if us_pron and us_pron.has_attr('data-src-mp3'):
        audio_url = us_pron['data-src-mp3']
    else:
        for div in soup.find_all('div', class_='sound'):
            if div.has_attr('data-src-mp3'):
                audio_url = div['data-src-mp3']
                if 'nam' in audio_url: break

    return definition_string, audio_url


# ─────────────────────────────────────────────
# 4. MAIN IMPORT LOGIC
# ─────────────────────────────────────────────
def main():
    oxford_words = extract_advanced_words(PDF_PATHS)
    invoke('createDeck', deck=DECK_NAME)

    added_count = 0
    skipped_count = 0

    print(f"\nStarting import to deck '{DECK_NAME}'...")

    for word, cefr_level in list(oxford_words.items())[:MAX_WORDS_TO_TEST]:

        if word_exists_in_anki(word):
            print(f"Skipping '{word}': Already exists in Anki.")
            skipped_count += 1
            continue

        print(f"Processing: {word} [{cefr_level}]")
        search_word = word.replace(" ", "-")

        # Fetch Data
        definition, audio_url = fetch_oxford_data(search_word)
        if definition is None:
            print(f"  -> '{word}' not found on Oxford. Skipping.")
            time.sleep(1)
            continue

        back_content = definition

        # Process Audio
        if audio_url:
            audio_res = requests.get(audio_url, headers=HEADERS)
            if audio_res.status_code == 200:
                audio_b64 = base64.b64encode(audio_res.content).decode('utf-8')
                safe_name = re.sub(r'[^\w\-]', '_', word)
                filename = f"oxford_import_{safe_name}.mp3"

                invoke('storeMediaFile', filename=filename, data=audio_b64)
                back_content += f"<br>[sound:{filename}]"

        # Create Note (Just Front and Back)
        note = {
            "deckName": DECK_NAME,
            "modelName": MODEL_NAME,
            "fields": {
                "Front": word,
                "Back": back_content
            },
            "options": {
                "allowDuplicate": False,
                "duplicateScope": "deck"
            },
            "tags": [
                "Oxford5000",
                cefr_level
            ]
        }

        try:
            invoke('addNote', note=note)
            print(f"  ✓ Added '{word}'")
            added_count += 1
        except Exception as e:
            print(f"  -> Failed to add '{word}' to Anki: {e}")

        # Polite delay for Oxford servers
        time.sleep(1.2)

    print("\n--- Import Complete ---")
    print(f"Successfully added: {added_count} cards")
    print(f"Skipped (Duplicates): {skipped_count} cards")


if __name__ == "__main__":
    main()