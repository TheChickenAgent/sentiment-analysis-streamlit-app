# python -m streamlit run streamlit_app.py in the right directory (cd C:\Users\MartijnElands\stack\UM DKE\BSc - Year 2\Period 2.1\KE@Work\GitRepo\lang-detection)

import json
import streamlit as st
from statistics import mode



# Method to see if a key is in the dictionary's frequency list
def keyInDictionaryFreq(dictionary, key):
    if key in dictionary.keys():
        return True
    else:
        return False

st.title("Language detection")
st.header("Welcome to this wonderful language detection app, powered by Streamlit.")

allList = [] # For the confusion matrix, input is expected and output what is predicted
languagesList = [] # List with all the languages that will be considered
lineCounter = 0 # Counts the number of lines


st.write("Please select the languages you would like to consider.")

eng = st.checkbox("English", True)
if eng:
    languagesList.append("en")

nld = st.checkbox("Dutch", True)
if nld:
    languagesList.append("nl")

ita = st.checkbox("Italian", True)
if ita:
    languagesList.append("it")

por = st.checkbox("Portugese", True)
if por:
    languagesList.append("pt")

fre = st.checkbox("French", True)
if fre:
    languagesList.append("fr")

deu = st.checkbox("German", True)
if deu:
    languagesList.append("de")

esp = st.checkbox("Spanish", True)
if esp:
    languagesList.append("es")

fin = st.checkbox("Finish", True)
if fin:
    languagesList.append("fi")

swe = st.checkbox("Swedish", True)
if swe:
    languagesList.append("sv")

#New languages that have been added
nor = st.checkbox("Norwegian", True)
if nor:
    languagesList.append("no")

rom = st.checkbox("Romanian", True)
if swe:
    languagesList.append("sv")

dan = st.checkbox("Danish", True)
if dan:
    languagesList.append("da")

ukr = st.checkbox("Ukrainian", True)
if ukr:
    languagesList.append("uk")

bug = st.checkbox("Bulgarian", True)
if bug:
    languagesList.append("bg")

hrz = st.checkbox("Croatian", True)
if hrz:
    languagesList.append("hr")

csz = st.checkbox("Czech", True)
if csz:
    languagesList.append("cs")

vit = st.checkbox("Vietnamese", True)
if vit:
    languagesList.append("vi")

rus = st.checkbox("Russian", True)
if rus:
    languagesList.append("ru")

zho = st.checkbox("Chinese", True)
if zho:
    languagesList.append("zh")

jap = st.checkbox("Japanese", True)
if jap:
    languagesList.append("ja")


# If we ingore the difference between upper and lower case letters,
# then we need to mutate the input dictionary to be only one type 
# of case letters. Therefore, make everything lower case.

languageFrequencies = {}
# Read frequencies from JSON files. _3 indicates it's a 3-gram
for language in languagesList:
    with open('frequency_sets_own/' + language + '_3.json') as f:
        languageFrequencies[language] = json.load(f)
f.close()

language_mapper = {}
language_mapper['en'] = 'English'
language_mapper['nl'] = 'Dutch'
language_mapper['it'] = 'Italian'
language_mapper['fr'] = 'French'
language_mapper['pt'] = 'Portugese'
language_mapper['de'] = 'German'
language_mapper['es'] = 'Spanish'
language_mapper['fi'] = 'Finish'
language_mapper['sv'] = 'Swedish'
language_mapper['no'] = 'Norwegian'

language_mapper['ro'] = 'Romanian'
language_mapper['da'] = 'Danish'
language_mapper['uk'] = 'Ukrainian'
language_mapper['bg'] = 'Bulgarian'
language_mapper['hr'] = 'Croatian'

language_mapper['cs'] = 'Czech'
language_mapper['vi'] = 'Vietnamese'
language_mapper['ru'] = 'Russian'
language_mapper['zh'] = 'Chinese'
language_mapper['ja'] = 'Japanese'

text = st.text_area("Put the text here. Each new line will be concidered as a new text.")

languageTriggers = dict.fromkeys(languagesList, 0)
text_key_counter = {}

pressed = st.button("Detect language")

if pressed & (text is not None):
    st.write("Processing...")
    st.write(" ") # Ensures nice spacing

    splitted = text.splitlines()

    for line in splitted:
        outputLanguage = [] # Everytime it will consider all languages equally
        lineCounter = lineCounter + 1 # For every line we have splitted, add 1 to the counter

        line = "  " + line # The first lettter is also a trigram, so add 2 spaces
        line = line + "  " # The last lettter is also a trigram, so add 2 spaces

        input_characters = ""
        for i in range(0, len(line)):
            if(line[i].isalpha() | line[i].isspace()):
                input_characters = input_characters + line[i]

        text_key_counter = {}
        for i in range(0, len(input_characters)-2):
            if keyInDictionaryFreq(text_key_counter, input_characters[i]+input_characters[i+1]+input_characters[i+2]):
                text_key_counter[input_characters[i]+input_characters[i+1]+input_characters[i+2]] = text_key_counter[input_characters[i]+input_characters[i+1]+input_characters[i+2]] + 1
            else:
                text_key_counter[input_characters[i]+input_characters[i+1]+input_characters[i+2]] = 1

        # Create dictionary for every language in languagesList and set its value to 0
        languageAbsoluteDistance = dict.fromkeys(languagesList, 0)
            
        # Split when there is a space, but ignore the first number because that's irrelevant for the program
        input_words = ( len(line.split()) - 1 )
            
        # Calculate the amount of characters, it has already removed whitespaces etc.
        input_characters_length = len(input_characters)

        # For every language, calculate the absolute distance between the UNION of sets and the input frequency
        for language in languagesList:
            keys = set(languageFrequencies[language].keys()).union(set(text_key_counter.keys()))
            for key in keys:
                languageAbsoluteDistance[language] = languageAbsoluteDistance[language] + abs(text_key_counter.get(key, 0)/input_characters_length - languageFrequencies[language].get(key, 0)/languageFrequencies[language]["characters"] )

        # Increase trigger based on the difference in distance.
        # append the index to see when it has guessed what for the confusion matrix
        min_key = min(languageAbsoluteDistance, key=languageAbsoluteDistance.get)
        languageTriggers[min_key] = languageTriggers[min_key] + 1
        outputLanguage.append(min_key)

        predicted_language = mode(outputLanguage)
        output_language = language_mapper[predicted_language]
        st.write("Processed and the result is:")
        st.write("Your text \"" + str(text) + "\" is most likely: " + output_language + ".") # working with trigrams, and the first two characters are spaces, so from 2:end should be in the list. +1 because Python indexes work from 0.
        st.write(" ") # Ensures nice spacing
        #splitted.index(line[2:]) + 1


expander = st.expander("More information")
expander.write("This language detection software is based on a N-gram model. It works with language frequencies that are based on [this website's datasets](https://wortschatz.uni-leipzig.de/en/download/).")
expander.write("This software is powered by [Streamlit](https://streamlit.io/).")
