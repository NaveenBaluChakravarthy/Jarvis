"""
Module: Main Driver
Project: Jarvis
Author: Naveen Chakravarthy Balasubramanian
"""

# Importing the libraries
import json
import random
import tkinter as tk
import tkinter.font as fnt
import tkinter.messagebox as msgbox
from answer_fetcher import AnswerFetcher as ans

# Load AnswerFetcher
ans = AnswerFetcher()

def winprotocol():
    """Window Protocol"""
    if msgbox.askyesno("Jarvis", "Do you really want to quit?"):
        msgbox.showinfo('Jarvis',
                        'The Chat History is saved as JarvisChatLog.txt in the same directory as this executable file.\n\nExcelsior!')
        window.destroy()


# Interaction - Send Click
def interact1():
    inp = ent_msg.get()
    if inp != '':
        inpt = inp.lower().split()
        inpt = ' '.join([word for word in inpt if word not in inpstd])
        basic = False
        for intent in data['intents']:
            for pattern in intent['patterns']:
                if pattern == inpt:
                    resps = list(intent['responses'])
                    basic = True
        txt_chat.configure(state='normal')
        userans = f'You    : {inp}\n'
        if basic:
            jar1ans = f'Jarvis : {random.choice(resps)}\n\n'
            txt_chat.insert(tk.END, userans, 'user')
            jarvislog.write(userans)
            txt_chat.insert(tk.END, jar1ans, 'jarvis')
            jarvislog.write(jar1ans)
        else:
            #returned_answer = "Hello"
            returned_answer = ans.generate_answer(inpt)
            jar2ans = f'Jarvis : {returned_answer}\n\n'
            txt_chat.insert(tk.END, userans, 'user')
            jarvislog.write(userans)
            txt_chat.insert(tk.END, jar2ans, 'jarvis')
            jarvislog.write(jar2ans)
        txt_chat.see(tk.END)
        ent_msg.delete(0, tk.END)
        txt_chat.configure(state='disabled')

# Interaction - Enter Keypress
def interact2(event):
    interact1()


# Load the json file
with open('D:\Code\Jarvis\intents.json') as file:
    data = json.loads(file.read())

# Initialize the Chat History Log
jarvislog = open('JarvisChatLog.txt', 'a')

# Constants
inpstd = ['jarvis', 'buddy', 'dude']

# Designing the interface
window = tk.Tk()
window.minsize(1024, 680)
window.maxsize(1024, 680)
window.title('Jarvis')
window.iconbitmap('D:\Code\Jarvis\Jarvis_ico.ico')
window.protocol("WM_DELETE_WINDOW", winprotocol)

# Designing the chat elements
jarvisfont = fnt.Font(family='Cambria', size=12)
lbl_status = tk.Label(window, text='Chat History with Jarvis', font=jarvisfont)
lbl_status.place(x=12, y=6)

txt_chat = tk.Text(window, bd=1, bg='white', width=985, height=580, font=jarvisfont)
txt_chat.place(x=12, y=35, width=985, height=580)

ent_msg = tk.Entry(window, bd=1, bg='white', width=800, font=jarvisfont)
ent_msg.place(x=12, y=630, width=800, height=35)
ent_msg.focus_set()
ent_msg.bind('<Return>', interact2)

btn_send = tk.Button(window, text='Send', command=interact1, background='green', activebackground='light green',  width=188, height=35)
btn_send.place(x=824, y=630, width=188, height=35)

scrollbar = tk.Scrollbar(window, command=txt_chat.yview)
scrollbar.place(x=1000, y=35, height=580)

txt_chat.configure(state='disabled', yscrollcommand=scrollbar.set)
txt_chat.tag_config('user', foreground='#9E0000')
txt_chat.tag_config('jarvis', foreground='#00009E')

# Executing the interface 
window.mainloop()

# Housekeeping
try:
    window.destroy()
except:
    pass

# Close the Chat History Log
jarvislog.close()
