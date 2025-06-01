import json
import re

def custom_parser(input_string):
    return input_string.replace("/$", "").replace("\\$", "")

def remove_special_chars(response_string):
    spe_char_to_remove = ['`', '`', 'json']
    for character in spe_char_to_remove:
        response_string = response_string.replace(character, '')
    return response_string


def extract_json_from_output(output):
    # Regex to find JSON enclosed in ```json ``` markers
    pattern = r'```json\n([\s\S]*?)\n```'
    match = re.search(pattern, output)

    if match:
        # Extracting the JSON string from the regex match
        json_string = match.group(1)

        # Correctly handling escape sequences
        # First, ensure backslashes are correctly interpreted
        json_string = json_string.replace('\\\\', '\\')
        # Then replace escaped double quotes
        json_string = json_string.replace('\\"', '"')

        # Converting the JSON string into a Python dictionary
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            raise e
    else:
        print("No JSON found")
        return None
