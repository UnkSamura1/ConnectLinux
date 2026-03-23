import json
import urllib.request
import requests
from bs4 import BeautifulSoup
import base64
import time

# --- Configuration ---
DECK_NAME = "English"  # Replace with your actual deck name
# Expanded headers to prevent Oxford from blocking the scraper
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}


# --- AnkiConnect Helper Functions ---
def request(action, **params):
    return {'action': action, 'params': params, 'version': 6}


def invoke(action, **params):
    requestJson = json.dumps(request(action, **params)).encode('utf-8')
    try:
        response = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8765', requestJson)))
    except Exception as e:
        raise Exception(f"Failed to connect to AnkiConnect. Is Anki running? Error: {e}")

    if response.get('error') is not None:
        raise Exception(response['error'])
    return response['result']


# --- Oxford Dictionary Scraper ---
def fetch_oxford_data(word):
    url = f"https://www.oxfordlearnersdictionaries.com/definition/english/{word}"
    res = requests.get(url, headers=HEADERS)

    # If the page doesn't exist (e.g., a 404 error), return None to trigger deletion
    if res.status_code != 200:
        return None, None

    soup = BeautifulSoup(res.content, 'html.parser')

    # 1. Extract All Definitions
    definitions = []
    # Oxford groups different senses/meanings under 'li' tags with class 'sense'
    senses = soup.find_all('li', class_='sense')

    for i, sense in enumerate(senses, 1):
        def_text_element = sense.find('span', class_='def')
        if def_text_element:
            # Format as "1. definition text"
            definitions.append(f"{i}. {def_text_element.text.strip()}")

    # Fallback in case the HTML structure slightly varies (no 'sense' class found)
    if not definitions:
        def_spans = soup.find_all('span', class_='def')
        for i, span in enumerate(def_spans, 1):
            definitions.append(f"{i}. {span.text.strip()}")

    # If we still found absolutely no definitions despite a 200 OK status, treat it as not found
    if not definitions:
        return None, None

    # Join all definitions with a line break for Anki formatting
    definition_string = "<br>".join(definitions)

    # 2. Extract Audio (Prioritizing US pronunciation)
    audio_url = None
    # Oxford uses divs with data-src-mp3 attributes for audio
    us_pron = soup.find('div', class_='pron-us')
    if us_pron and us_pron.has_attr('data-src-mp3'):
        audio_url = us_pron['data-src-mp3']
    else:
        # Fallback: Find any sound div and check for mp3
        sound_divs = soup.find_all('div', class_='sound')
        for div in sound_divs:
            if div.has_attr('data-src-mp3'):
                audio_url = div['data-src-mp3']
                # Try to grab the North American ('nam') version if available
                if 'nam' in audio_url:
                    break

    return definition_string, audio_url


# --- Main Logic ---
def main():
    print(f"Searching for empty cards in deck: '{DECK_NAME}'...")

    # Find note IDs where the "Back" field is empty
    query = f'"deck:{DECK_NAME}" "Back:"'
    note_ids = invoke('findNotes', query=query)

    if not note_ids:
        print("No empty cards found. Everything is up to date!")
        return

    notes_info = invoke('notesInfo', notes=note_ids)

    for note in notes_info:
        note_id = note['noteId']
        # Extract the word from the front field, stripping any HTML tags Anki might add
        front_text = BeautifulSoup(note['fields']['Front']['value'], "html.parser").text.strip()

        print(f"\nProcessing: {front_text}")

        # Format word for URL (e.g., "apple tree" -> "apple-tree")
        search_word = front_text.lower().replace(" ", "-")
        definition, audio_url = fetch_oxford_data(search_word)

        # --- Delete card if word is not found ---
        if definition is None:
            print(f"Word '{front_text}' not found on Oxford (or has no definitions). Deleting card...")
            invoke('deleteNotes', notes=[note_id])
            time.sleep(1)  # Be polite to servers even on a failure
            continue
        # ----------------------------------------

        back_content = definition

        # Download and store audio if found
        if audio_url:
            print(f"Downloading audio...")
            audio_res = requests.get(audio_url, headers=HEADERS)
            if audio_res.status_code == 200:
                audio_b64 = base64.b64encode(audio_res.content).decode('utf-8')
                # Clean up filename to prevent Anki media errors
                safe_filename = front_text.replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = f"oxford_{safe_filename}.mp3"

                # Send the media file to Anki's database
                invoke('storeMediaFile', filename=filename, data=audio_b64)

                # Append the Anki sound tag to the back field
                back_content += f"<br><br>[sound:{filename}]"
            else:
                print("Failed to download audio file.")
        else:
            print("No audio URL found on Oxford Dictionary.")

        # Update the note in Anki
        invoke('updateNoteFields', note={
            'id': note_id,
            'fields': {
                'Back': back_content
            }
        })
        print(f"Successfully updated '{front_text}'!")

        # Be polite to Oxford Dictionary servers to avoid IP bans
        time.sleep(1)


if __name__ == "__main__":
    main()