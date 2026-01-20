import json
import os
import requests


MEDIATOPICS_URL = "https://cv.iptc.org/newscodes/mediatopic?lang=en-US&format=json"

MEDIATOPICS_PATH = "./schema/mediatopic_cptall-en-US.json"


# Function to download the Media Topics JSON file
def download_mediatopics_json():
    try:
        # request media topics
        response = requests.get(MEDIATOPICS_URL)
        # check if the request was successful
        response.raise_for_status()
        # parse the JSON content into a dictionary
        data = response.json()
        # create the schema directory if it doesn't exist
        os.makedirs("./schema", exist_ok=True)
        # write the data to the JSON file
        with open(MEDIATOPICS_PATH, 'w') as f:
            json.dump(data, f, indent=4)

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")


def format_broad_topics():
    concepts_dict = {concept['qcode']: concept for concept in media_topics['conceptSet'] if
                     'retired' not in concept}
    res = {}
    for broad_topic_uri in media_topics.get('hasTopConcept', []):
        qcode = f"medtop:{broad_topic_uri.split('/')[-1]}"
        concept = concepts_dict.get(qcode, {})
        res[concept.get('prefLabel', {}).get('en-US')] = concept.get('definition', {}).get('en-US')

    return json.dumps(res)


# Download the Media Topics JSON file if it doesn't exist
if not os.path.exists(MEDIATOPICS_PATH):
    print("Downloading Media Topics Controlled Vocabulary from IPTC web")
    download_mediatopics_json()
# Load the Media Topics JSON file
with open(MEDIATOPICS_PATH, "r") as file:
    media_topics = json.load(file)
    broad_topics_json = format_broad_topics()
