import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import subprocess

def sttr():
    subprocess.Popen(['python3', '../scripts/train_model.py'])

def stri():
    subprocess.Popen(['python3', '../scripts/real_time_inference.py'])

def ort():
    log_dir = "../logs/fit/"
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

root = tk.Tk()
root.title('Captcha Transformice Safemarket')

f = tk.Frame(root)
f.pack(pady=20)

t = tk.Button(f, text='Start Training', command=sttr, width=20)
t.pack(pady=10)

i = tk.Button(f, text='Start Real-Time Inference', command=stri, width=30)
i.pack(pady=10)

tb = tk.Button(f, text='Open TensorBoard', command=ort, width=20)
tb.pack(pady=10)

root.mainloop()
