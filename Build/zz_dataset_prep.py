"""
Module: Dataset Preparation 01
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""

# Importing the libraries
import pandas as pd
import json

# Function definitions for question answer pair generation


def finish(q, a):
    qnalist.append((q, a))


def qa_rel_sub(qtype, ph_rel, chara, charb):
    if qtype == "fm":
        ph_movie = f"In the movie {players['Movie'][i]}, "
        ph_sub = chara
        ph_obj = charb
    elif qtype == 'bm':
        ph_movie = f"In the movie {players['Movie'][i]}, "
        ph_sub = charb
        ph_obj = chara
    elif qtype == 'f0':
        ph_movie = ''
        ph_sub = chara
        ph_obj = charb
    elif qtype == 'b0':
        ph_movie = ''
        ph_sub = charb
        ph_obj = chara
    else:
        pass
    q = f"{ph_movie}Who is the {ph_rel} of {ph_obj}?"
    a = f"{ph_movie}{ph_sub} is the {ph_rel} of {ph_obj}."
    finish(q, a)
    if ph_rel in ['father', 'mother']:
        qa_rel_sub(qtype, 'parent', chara, charb)
    elif ph_rel in ['son', 'daughter']:
        qa_rel_sub(qtype, 'child', chara, charb)


def qa_relations():
    for i in range(len(players)):
        chara = players['Char A'][i]
        charb = players['Char B'][i]
        relations = str(players['Relation'][i]).split('/')
        if len(relations) == 2:
            qa_rel_sub('fm', relations[0], chara, charb)
            qa_rel_sub('f0', relations[0], chara, charb)
            qa_rel_sub('bm', relations[1], chara, charb)
            qa_rel_sub('b0', relations[1], chara, charb)
        elif len(relations) == 1:
            qa_rel_sub('fm', relations[0], chara, charb)
            qa_rel_sub('f0', relations[0], chara, charb)
        else:
            print('Problem')


def qa_names():
    q_format = ["Who is", "What is the birth name of",
                "What is the formal name of", "What is the real name of"
                ]
    for key, value in alias.items():
        valuex = [aa for aa in value if aa != 'nan']
        valuex.append(key)
        for qf in q_format:
            if len(valuex) > 1:
                ph_aka = f", also known as {', '.join(valuex)}. "
            else:
                ph_aka = ''
            if gender[key] == 'Masculine':
                ph_aka += 'He is'
            elif gender[key] == 'Feminine':
                ph_aka += 'She is'
            elif gender[key] == 'NeuterSingular':
                ph_aka += 'It is'
            elif gender[key] == 'NeuterPlural':
                ph_aka += 'Those are'
            elif gender[key] == 'Plural':
                ph_aka += 'They are'
            else:
                pass
            q = f"{qf} {key}?"
            a = f"{key}, of formal name {valuex[-2]}{ph_aka} a {', '.join(persona[key])}."
            finish(q, a)


def qa_actors():
    q_format = ["Who played the character", "Who portrayed"]
    for qf in q_format:
        for castkey, castvalue in cast.items():
            if castvalue != 'nan':
                q = f"{qf} {castkey}?"
                a = f"The character {castkey} was portrayed by {castvalue}."
                finish(q, a)
    q_format = ["Which character was played by", "Which character was portrayed by"]
    for qf in q_format:
        for castkey, castvalue in cast.items():
            if castvalue != 'nan':
                q = f"{qf} {castvalue}?"
                a = f"The character {castkey} was portrayed by {castvalue}."
                finish(q, a)
                q = f"{qf} {castvalue}?"
                a = f"The character {castkey} was portrayed by {castvalue}."
                finish(q, a)


def qa_tracks():
    q_format = ["Which song plays in the background of",
                "Which soundtrack plays in",
                "What is the background music in the"]
    for i in range(len(music)):
        for qf in q_format:
            q = f"{qf} {music['Scene'][i]} scene in the movie {music['Movie'][i]}?"
            a = f"The soundtrack {music['Soundtrack'][i]} plays in the {music['Scene'][i]} scene in the movie {music['Movie'][i]}"
            finish(q, a)


def qa_locations():
    import random
    zz = [random.choice(['Event', 'Place']) for i in range(len(locations))]
    locations['Type'] = zz
    for i in range(len(locations)):
        if locations['Type'][i] == 'Event':
            q = f"In the movie {locations['Movie'][i]}, where does the {locations['Scene'][i]} scene take place?"
            a = f"In the movie {locations['Movie'][i]}, the {locations['Scene'][i]} scene takes place in {locations['Location'][i]}."
        elif locations['Type'][i] == 'Place':
            q = f"In the movie {locations['Movie'][i]}, where is the {locations['Scene'][i]} situated?"
            a = f"In the movie {locations['Movie'][i]}, the {locations['Scene'][i]} is situated in {locations['Location'][i]}."
        else:
            pass
        finish(q, a)


def qa_dialogues():
    for i in range(len(dialogues)):
        q = f"To whom did {dialogues['Speaker'][i]} tell this - {dialogues['Dialogue'][i]} - in the movie {dialogues['Movie'][i]} about {dialogues['Context'][i]} in the {dialogues['Scene'][i]} scene?"
        a = f"{dialogues['Receiver'][i]}"
        finish(q, a)
        q = f"What did {dialogues['Speaker'][i]} say to {dialogues['Receiver'][i]} in the {dialogues['Scene'][i]} scene in the movie {dialogues['Movie'][i]} about {dialogues['Context'][i]}?"
        a = f"{dialogues['Dialogue'][i]}"
        finish(q, a)
        q = f"In which movie did {dialogues['Speaker'][i]} tell this - {dialogues['Dialogue'][i]} - to {dialogues['Receiver'][i]} about {dialogues['Context'][i]} in the {dialogues['Scene'][i]} scene?"
        a = f"{dialogues['Movie'][i]}"
        finish(q, a)
        q = f"In which scene of the movie {dialogues['Movie'][i]}, did {dialogues['Speaker'][i]} tell this - {dialogues['Dialogue'][i]} - to {dialogues['Receiver'][i]} about {dialogues['Context'][i]}?"
        a = f"{dialogues['Scene'][i]}"
        finish(q, a)
        q = f"Who said this - {dialogues['Dialogue'][i]} - to {dialogues['Receiver'][i]} in the {dialogues['Scene'][i]} scene of the movie {dialogues['Movie'][i]} about {dialogues['Context'][i]}?"
        a = f"{dialogues['Speaker'][i]}"
        finish(q, a)
        q = f"About what did {dialogues['Speaker'][i]} say this - {dialogues['Dialogue'][i]} - to {dialogues['Receiver'][i]} in the {dialogues['Scene'][i]} scene of the movie {dialogues['Movie'][i]}?"
        a = f"{dialogues['Context'][i]}"
        finish(q, a)


def qa_happenings():
    for i in range(len(happenings)):
        q = f"In the movie {happenings['Movie'][i]}, when did {happenings['Character'][i]} {happenings['Event'][i]}?"
        a = f"{happenings['Time'][i]}"
        finish(q, a)


def qa_misc():
    for i in range(len(misc)):
        q = f"In the movie {misc['Movie'][i]}, {misc['Question'][i]}"
        a = misc['Answer'][i]
        finish(q, a)


# Getting the raw data
workbook_name = 'Marvel v1.x.xlsx'
players = pd.read_excel(workbook_name, sheet_name='Players')
aliases = pd.read_excel(workbook_name, sheet_name='Aliases')
music = pd.read_excel(workbook_name, sheet_name='Music')
locations = pd.read_excel(workbook_name, sheet_name='Locations')
dialogues = pd.read_excel(workbook_name, sheet_name='Dialogues')
happenings = pd.read_excel(workbook_name, sheet_name='Happenings')
misc = pd.read_excel(workbook_name, sheet_name='Misc')

# Initializing the variables
qnalist = []
alias = {}
persona = {}
gender = {}
cast = {}

# Setting things up
for i in range(len(aliases)):
    key = aliases['Common Name'][i]
    value = str(aliases['AKA'][i]).split(', ')
    value.append(aliases['Formal Name'][i])
    alias[key] = value
    persona[key] = str(aliases['Persona'][i]).split(', ')
    gender[key] = str(aliases['Gender'][i])
    cast[key] = str(aliases['Actor'][i])

# QA Generation Function Calls
qa_relations()
qa_names()
qa_actors()
qa_tracks()
qa_locations()
qa_dialogues()
qa_happenings()
qa_misc()

# Json Serialization
qna_serialized = {"Data": qnalist}
with open("data.json", "w") as jsonfile:
    json.dump(qna_serialized, jsonfile)
