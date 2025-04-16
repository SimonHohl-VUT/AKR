from imgui_bundle import imgui, implot, immapp, ImVec2
import time
import numpy as np
from threading import Thread
import rsa
import os

import trial
import dixon
import cfrac
import qsieve

funs = [
    [False, "TD", "Trial division", trial.fun],
    [False, "DF", "Dixon's factorization method", dixon.fun],
    [False, "CF", "Continued fraction factorization", cfrac.fun],
    [False, "QS", "Quadratic sieve", qsieve.fun],
]

running = False

times = np.array([0.0] * len(funs), dtype = np.double)
labels = []

for fun in funs:
    labels.append(fun[1])

nBits = "128"

log = ""

class ReturnThread(Thread):
    """
    A custom thread class to handle factorization in a separate thread.
    """

    def __init__(self, group = None, target = None, name = None, args = (), kwargs = {}):
        super().__init__(group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        super().join()
        return self._return

def run():
    """
    The main function to run the factorization methods and log the results.
    """

    global running
    global log
    global nBits
    global times

    for i in range(len(funs)):
        times[i] = 0.0

    log += "============================\nStarting execution\n"

    log += "Generating RSA keys: {} bits\n".format(nBits)
    bits = int(nBits)
    public, private = rsa.newkeys(bits)

    data = b""
    cipher = b""
    if bits >= 90:
        data = os.urandom(rsa.common.byte_size(public.n) - 11)
        cipher = rsa.encrypt(data, public)
    else:
        if private.p > private.q:
            private = (private.q, private.p)
        else:
            private = (private.p, private.q)

    for i in range(len(funs)):
        fun = funs[i]

        if not fun[0]:
            continue

        log += "Starting {}\n".format(fun[2])

        t = ReturnThread(target = fun[3], args = (public.n,))
        start = time.time()
        t.start()
        while t.is_alive():
            times[i] = time.time() - start
        p, q = t.join()
        times[i] = time.time() - start

        log += "{} finished in {} seconds\n".format(fun[2], times[i])

        if bits >= 90:
            _, d = rsa.key.calculate_keys_custom_exponent(p, q, public.e)
            private = rsa.PrivateKey(public.n, public.e, d, p, q)
            decrypted = rsa.decrypt(cipher, private)

            if data == decrypted:
                log += "Private key was found successfully\n"
            else:
                log += "Failed to find private key\n"
        else:
            if (p, q) == private:
                log += "Private key was found successfully\n"
            else:
                log += "Failed to find private key\n"

    log += "Execution finished\n"
    running = False

def loop():
    """
    The main loop for the GUI, handling user input and displaying results.
    """

    global running
    global funs
    global times
    global labels
    global nBits
    global log

    imgui.begin("RSA-CRACK")
    _, value = imgui.input_text("Number of bits to generate", nBits)
    if not running:
        try:
            if int(value) < 16:
                nBits = "16"
            else:
                nBits = value
        except:
            pass
    for fun in funs:
        _, value = imgui.checkbox(fun[2], fun[0])
        if not running:
            fun[0] = value
    if not running and imgui.button("Start"):
        running = True
        Thread(target = run).start()
    imgui.input_text_multiline("##log", log, ImVec2(-1, -1), imgui.InputTextFlags_.read_only.value)
    imgui.end()

    imgui.begin("Times" + (" - running" if running else "") + "###timeswindow")
    if implot.begin_plot("##timesplot"):
        implot.setup_axes("Algorithm", "Time [s]", implot.AxisFlags_.auto_fit.value, implot.AxisFlags_.auto_fit.value)
        implot.setup_axis_ticks(axis = implot.ImAxis_.x1.value, v_min = 0, v_max = len(funs) - 1, n_ticks = len(funs), labels = labels, keep_default = False)
        implot.plot_bar_groups(label_ids = ["##time"], values = times)
        implot.end_plot()
    
    text = ""
    for i in range(len(funs)):
        fun = funs[i]
        text += "{} - {}: {}{}".format(fun[1], fun[2], times[i], "\n" if i != (len(funs) - 1) else "")
    imgui.input_text_multiline("##timestext", text, ImVec2(-1, -1), imgui.InputTextFlags_.read_only.value)
    imgui.end()

immapp.run(loop, with_implot = True, fps_idle = 0)